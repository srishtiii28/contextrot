import time
from fastapi import HTTPException
from sqlalchemy import select
from app.core.config import get_settings
from app.database.models import Constraint, Message, StudySession
from app.database.runtime_models import Event, Job, MessageState
from app.integrations.rag_client import RetrievalRequest, retrieve
from app.schemas.analysis import AnalysisInput
from app.schemas.message import MessageResponse
from app.services.llm_service import compress_history


def serialize_message(db, message):
    result = MessageResponse.model_validate(message)
    detail = db.get(MessageState, message.id)
    if detail:
        result.status, result.sources = detail.status, detail.sources
        result.correction_of, result.generation_ms = detail.correction_of, detail.generation_ms
        result.scope_warning = getattr(detail, "scope_warning", None)
    return result


def event(db, session_id, kind, payload):
    db.add(Event(session_id=session_id, kind=kind, payload=payload))


async def prepare(db, session_id, payload):
    settings = get_settings()
    if payload.demo_fault and (not settings.enable_demo_faults or settings.llm_provider not in {"offline", "mock"}):
        raise HTTPException(403, "Fault injection is enabled only in explicit offline demo mode")
    if db.scalar(select(Job.id).where(Job.session_id == session_id, Job.status.in_(["queued", "running"]))):
        raise HTTPException(409, "Wait for the previous answer's reliability check to complete")
    history = [dict(role=m.role, content=m.content) for m in db.scalars(select(Message).where(Message.session_id == session_id).order_by(Message.id))]
    constraints = [dict(type=c.constraint_type, content=c.content) for c in db.scalars(select(Constraint).where(Constraint.session_id == session_id))]
    session = db.get(StudySession, session_id)
    kb_id = session.knowledge_base_id if session else None

    from app.services.rag_service import (
        is_follow_up_request,
        is_pure_continuation_request,
        normalize_query_typos,
        resolve_conversational_query,
    )

    cleaned_query = normalize_query_typos(payload.content)
    is_continuation = is_pure_continuation_request(cleaned_query) or is_follow_up_request(cleaned_query)
    if is_continuation and history:
        prev_in_scope = True
        last_assistant = next((h for h in reversed(history) if h["role"] == "assistant"), None)
        if last_assistant:
            content = last_assistant.get("content", "")
            if "> ⚠️ **out of syllabus:**" in content.lower() or "outside your prescribed course material" in content.lower():
                prev_in_scope = False

        if prev_in_scope:
            _, retrieval_query = resolve_conversational_query(cleaned_query, history)
            retrieved = await retrieve(RetrievalRequest(query=retrieval_query, session_id=session_id, knowledge_base_id=kb_id), db=db)
            chunks = retrieved.chunks
            if not chunks:
                substantive_topics = [
                    h["content"] for h in history if h.get("role") == "user"
                    and not (is_pure_continuation_request(h.get("content", "")) or is_follow_up_request(h.get("content", "")))
                ]
                if substantive_topics:
                    fb_retrieved = await retrieve(RetrievalRequest(query=substantive_topics[-1], session_id=session_id, knowledge_base_id=kb_id), db=db)
                    chunks = fb_retrieved.chunks
            return history, constraints, chunks, True
        else:
            return history, constraints, [], False

    retrieved = await retrieve(RetrievalRequest(query=cleaned_query, session_id=session_id, knowledge_base_id=kb_id), db=db)
    return history, constraints, retrieved.chunks, retrieved.in_scope


def inject_fault(answer, fault):
    if fault in {"contradiction", "persistent"}:
        return "A primary key can contain NULL values. A primary key can contain duplicate values."
    if fault == "instruction":
        return "A primary key uniquely identifies a row. It cannot contain NULL. It cannot contain duplicates. It enforces identity. It is chosen from candidate keys."
    if fault == "drift":
        return "Photosynthesis converts light into chemical energy in plants."
    return answer


def persist_answer(db, session_id, payload, answer, history, constraints, chunks, elapsed_ms, diagrams=None, in_scope=None):
    if in_scope is None:
        in_scope = bool(chunks)
    user_message = Message(session_id=session_id, role="user", content=payload.content, image_data=payload.image_data)
    assistant = Message(session_id=session_id, role="assistant", content=answer, agent_type="tutor" if in_scope else "scope_filter")
    db.add_all([user_message, assistant])
    db.flush()
    sources = [c.model_dump() for c in chunks]
    scope_warning = (
        "⚠️ This question is outside your course syllabus. The following answer is generated from general AI knowledge."
        if not in_scope
        else None
    )
    db.add(MessageState(
        message_id=assistant.id,
        status="pending" if in_scope else "refused",
        sources=sources,
        generation_ms=elapsed_ms,
        scope_warning=scope_warning,
    ))
    job = None
    if in_scope:
        analysis = AnalysisInput(
            session_id=session_id,
            message_id=assistant.id,
            current_question=payload.content,
            current_answer=answer,
            conversation_history=compress_history(history),
            constraints=constraints,
            retrieved_context=sources,
        )
        job = Job(
            session_id=session_id,
            message_id=assistant.id,
            payload={"analysis": analysis.model_dump(), "persistent_fault": payload.demo_fault == "persistent", "demo_fault": payload.demo_fault},
        )
        db.add(job)
        db.flush()
    event(db, session_id, "answer.saved", {"message_id": assistant.id, "job_id": job.id if job else None, "demo_fault": payload.demo_fault})
    db.commit()
    formatted_sources = [{"source": c.source, "page": c.page, "score": round(c.score, 4)} for c in chunks]
    return dict(
        user_message=serialize_message(db, user_message),
        assistant_message=serialize_message(db, assistant),
        job_id=job.id if job else None,
        sources=formatted_sources,
        diagrams=diagrams or [],
        scope_warning=scope_warning,
    )
