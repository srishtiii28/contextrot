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
from app.database.models import AnalysisResult, Message
from app.database.runtime_models import Job, MessageState
from app.integrations.rag_client import RetrievalRequest, retrieve
from app.integrations.rot_client import analyze
from app.schemas.analysis import AnalysisInput
from app.services.llm_service import compress_history, generate_response
from app.services.pipeline import event, inject_fault

log = logging.getLogger(__name__)


def claim_job():
    now, token = time.time(), uuid4().hex
    eligible = or_(Job.status == "queued", and_(Job.status == "running", Job.lease_until < now))
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
            retrieval = await retrieve(RetrievalRequest(query=payload.current_question, session_id=payload.session_id))
            sources = [c.model_dump() for c in retrieval.chunks]
            corrected = await generate_response(user_message=payload.current_question, conversation_history=compress_history(payload.conversation_history,3000),
                retrieved_context=retrieval.chunks, constraints=payload.constraints, recovery=True)
            if snapshot.get("persistent_fault"): corrected = inject_fault(corrected,"persistent")
            corrected_result = await analyze(payload.model_copy(update={"current_answer":corrected, "retrieved_context":sources}))
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
                status = "corrected" if corrected_result["health_score"] >= get_settings().recovery_threshold else "flagged"
                db.add(MessageState(message_id=message.id,status=status,sources=sources,correction_of=payload.message_id))
                db.add(AnalysisResult(session_id=payload.session_id,message_id=message.id,**corrected_result))
                event(db,payload.session_id,"answer.corrected",dict(message_id=message.id,correction_of=payload.message_id,status=status,**corrected_result))
            db.commit()
    except asyncio.CancelledError:
        with SessionLocal() as db:
            db.execute(update(Job).where(Job.id==job_id,Job.claim==token,Job.status=="running").values(status="queued",claim=None,lease_until=0));db.commit()
        raise
    except Exception as exc:
        log.warning("Reliability job %s failed: %s",job_id,type(exc).__name__)
        with SessionLocal() as db:
            locked=db.execute(update(Job).where(Job.id==job_id,Job.claim==token,Job.status=="running").values(status="failed",error=type(exc).__name__+": check worker/service configuration",finished_at=datetime.now(UTC)))
            if locked.rowcount:
                job=db.get(Job,job_id);db.get(MessageState,job.message_id).status="check_failed"
                event(db,job.session_id,"analysis.failed",dict(job_id=job_id,message_id=job.message_id,error="Reliability service failed; answer is unverified"))
            db.commit()
    return True


async def run_worker():
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
