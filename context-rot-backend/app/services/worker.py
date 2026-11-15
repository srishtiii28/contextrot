"""Database-polled worker, usable embedded or as `python -m app.services.worker`.
Atomic leases + fenced finalization allow restart/reclaim without duplicate corrections.
"""
import asyncio
import logging
import time
from datetime import datetime, UTC
from uuid import uuid4
from sqlalchemy import and_, or_, select, update
from app.core.config import get_settings
from app.database.database import SessionLocal, create_tables
from app.database.models import AnalysisResult, Message, StudySession
from app.database.runtime_models import Job, MessageState
from app.integrations.rag_client import RetrievalRequest, retrieve
from app.integrations.rot_client import analyze
from app.schemas.analysis import AnalysisInput
from app.services.llm_service import compress_history, generate_response
from app.services.pipeline import event, inject_fault

log = logging.getLogger(__name__)


MAX_RATE_LIMIT_REQUEUES = 20
MAX_REQUEUE_WAIT_SECONDS = 60
MAX_TIMEOUT_REQUEUES = 1


def _transient_wait(exc) -> tuple[float | None, str, int]:
    """(seconds to wait, counter name, cap) for temporary provider failures; (None, "", 0) otherwise."""
    import httpx
    if getattr(exc, "retry_after", None) is not None:
        return float(exc.retry_after), "rate_limited", MAX_RATE_LIMIT_REQUEUES
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        from app.reliability.judge_client import rate_limit_wait
        return rate_limit_wait(exc.response), "rate_limited", MAX_RATE_LIMIT_REQUEUES
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException)):
        return 2.0, "timed_out", MAX_TIMEOUT_REQUEUES
    return None, "", 0


async def _check_correction(payload, max_wait=30.0):
    """Scores the regenerated answer. Short rate limits and one timeout are waited out here: re-queuing the
    whole job would regenerate the correction and spend tutor tokens again. Returns None if it cannot be scored,
    and the correction is then saved as unverified while the original answer keeps its score."""
    waited, timeouts = 0.0, 0
    while True:
        try:
            return await analyze(payload)
        except Exception as exc:
            wait, kind, _ = _transient_wait(exc)
            if kind == "rate_limited" and waited + wait <= max_wait:
                waited += wait
                await asyncio.sleep(wait)
                continue
            if kind == "timed_out" and timeouts < MAX_TIMEOUT_REQUEUES:
                timeouts += 1
                continue
            log.warning("Reliability check of correction for message %s failed: %s: %s", payload.message_id, type(exc).__name__, exc)
            return None


def claim_job():
    now, token = time.time(), uuid4().hex
    # queued jobs re-queued after a rate limit carry a not-before time in lease_until
    eligible = or_(and_(Job.status == "queued", Job.lease_until <= now), and_(Job.status == "running", Job.lease_until < now))
    with SessionLocal() as db:
        candidate = db.scalar(select(Job.id).where(eligible).order_by(Job.id).limit(1))
        if candidate is None: return None
        result = db.execute(update(Job).where(Job.id == candidate, eligible).values(status="running", claim=token,
            lease_until=now+get_settings().job_lease_seconds, attempts=Job.attempts+1))
        db.commit()
        return (candidate, token) if result.rowcount else None


async def process_one():
    claimed = claim_job()
    if not claimed: return False
    job_id, token = claimed
    with SessionLocal() as db:
        job = db.get(Job,job_id)
        snapshot = job.payload
        attempts = job.attempts
    try:
        if attempts > 3: raise RuntimeError("Job exceeded three processing attempts")
        payload = AnalysisInput.model_validate(snapshot["analysis"])
        result = await analyze(payload)
        corrected, corrected_result, sources = None, None, []
        if result["health_score"] < get_settings().recovery_threshold:
            # Re-retrieve from the session's own course library; without knowledge_base_id, retrieval falls back
            # to the built-in demo notes and the correction is grounded in the wrong course.
            with SessionLocal() as db:
                study_session = db.get(StudySession, payload.session_id)
                kb_id = study_session.knowledge_base_id if study_session else None
            retrieval = await retrieve(RetrievalRequest(query=payload.current_question, session_id=payload.session_id, knowledge_base_id=kb_id))
            sources = [c.model_dump() for c in retrieval.chunks]
            corrected = await generate_response(user_message=payload.current_question, conversation_history=compress_history(payload.conversation_history,3000),
                retrieved_context=retrieval.chunks, constraints=payload.constraints, recovery=True, recovery_feedback=result.get("alert"))
            if snapshot.get("persistent_fault"): corrected = inject_fault(corrected,"persistent")
            corrected_result = await _check_correction(payload.model_copy(update={"current_answer":corrected, "retrieved_context":sources}))
        with SessionLocal() as db:
            # Acquire a write lock and fence stale workers before writing any result.
            locked = db.execute(update(Job).where(Job.id==job_id, Job.claim==token, Job.status=="running").values(status="done", finished_at=datetime.now(UTC)))
            if not locked.rowcount: db.rollback(); return True
            original = db.get(MessageState,payload.message_id)
            original.status = "superseded" if corrected is not None else "checked"
            db.add(AnalysisResult(session_id=payload.session_id,message_id=payload.message_id,**result))
            event(db,payload.session_id,"analysis.completed",dict(message_id=payload.message_id,**result))
            if corrected is not None:
                message = Message(session_id=payload.session_id,role="assistant",content=corrected,agent_type="correction")
                db.add(message);db.flush()
                if corrected_result is None:
                    status = "check_failed"
                elif corrected_result["health_score"] >= get_settings().recovery_threshold:
                    status = "corrected"
                else:
                    status = "flagged"
                db.add(MessageState(message_id=message.id,status=status,sources=sources,correction_of=payload.message_id))
                if corrected_result is not None:
                    db.add(AnalysisResult(session_id=payload.session_id,message_id=message.id,**corrected_result))
                event(db,payload.session_id,"answer.corrected",dict(message_id=message.id,correction_of=payload.message_id,status=status,**(corrected_result or {})))
            db.commit()
    except asyncio.CancelledError:
        with SessionLocal() as db:
            db.execute(update(Job).where(Job.id==job_id,Job.claim==token,Job.status=="running").values(status="queued",claim=None,lease_until=0));db.commit()
        raise
    except Exception as exc:
        wait, kind, cap = _transient_wait(exc)
        # A pending check blocks the student's next question, so long provider back-offs fail visibly instead
        # (the answer is marked unverified and can be retried with POST /sessions/{id}/jobs/{job_id}/retry).
        if wait is not None and wait <= MAX_REQUEUE_WAIT_SECONDS and snapshot.get(kind, 0) < cap:
            # Rate limits and provider timeouts are temporary: check again shortly instead of marking the answer unverified.
            log.warning("Reliability job %s %s; re-queued for %.1fs", job_id, kind.replace("_", " "), wait)
            with SessionLocal() as db:
                db.execute(update(Job).where(Job.id==job_id,Job.claim==token,Job.status=="running").values(
                    status="queued",claim=None,lease_until=time.time()+wait,attempts=Job.attempts-1,
                    payload=dict(snapshot, **{kind: snapshot.get(kind, 0)+1})))
                db.commit()
            return True
        log.warning("Reliability job %s failed: %s: %s",job_id,type(exc).__name__,exc)
        with SessionLocal() as db:
            locked=db.execute(update(Job).where(Job.id==job_id,Job.claim==token,Job.status=="running").values(status="failed",error=type(exc).__name__+": check worker/service configuration",finished_at=datetime.now(UTC)))
            if locked.rowcount:
                job=db.get(Job,job_id);db.get(MessageState,job.message_id).status="check_failed"
                event(db,job.session_id,"analysis.failed",dict(job_id=job_id,message_id=job.message_id,error="Reliability service failed; answer is unverified"))
            db.commit()
    return True


async def run_worker():
    if get_settings().reliability_detector == "nli" and not get_settings().rot_service_url:
        # Load the NLI models before the first job so model loading does not count against the scoring timeout.
        from app.reliability import nli
        try:
            await asyncio.to_thread(nli.get_backend)
            log.info("Reliability v2 models loaded (%s, %s)", get_settings().nli_model, get_settings().embedding_model)
        except Exception:
            # Keep polling: each job will then fail visibly as check_failed instead of hanging in the queue.
            log.exception("Could not load reliability v2 models; answers will be marked check_failed")
    while True:
        try:
            worked = await process_one()
            await asyncio.sleep(0 if worked else get_settings().worker_poll_seconds)
        except asyncio.CancelledError: raise
        except Exception:
            log.exception("Worker polling failed")
            await asyncio.sleep(1)


if __name__ == "__main__":
    create_tables()
    asyncio.run(run_worker())
