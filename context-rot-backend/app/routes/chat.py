from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import AnalysisResult, Constraint, Message, User
from app.integrations.rag_client import RetrievalRequest, retrieve
from app.integrations.rot_client import analyze
from app.routes.dependencies import current_user
from app.schemas.analysis import AnalysisInput
from app.schemas.message import ChatResponse, MessageCreate
from app.services.llm_service import generate_response
from app.services.session_service import owned_session

router = APIRouter(prefix="/sessions", tags=["chat"])


@router.post("/{session_id}/messages", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
async def send_message(session_id: int, payload: MessageCreate, db: Session = Depends(get_db), user: User = Depends(current_user)) -> ChatResponse:
    owned_session(db, session_id=session_id, user_id=user.id)
    user_message = Message(session_id=session_id, role="user", content=payload.content)
    db.add(user_message)
    db.flush()
    history = list(db.scalars(select(Message).where(Message.session_id == session_id).order_by(Message.created_at, Message.id)))
    try:
        retrieved = await retrieve(RetrievalRequest(query=payload.content, session_id=session_id))
        answer = await generate_response(
            user_message=payload.content,
            conversation_history=[{"role": item.role, "content": item.content} for item in history],
            retrieved_context=retrieved.chunks,
        )
    except NotImplementedError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    assistant_message = Message(session_id=session_id, role="assistant", content=answer, agent_type="llm")
    db.add(assistant_message)
    db.commit()
    db.refresh(user_message)
    db.refresh(assistant_message)

    constraints = list(db.scalars(select(Constraint).where(Constraint.session_id == session_id)))
    analysis_payload = AnalysisInput(
        session_id=session_id,
        message_id=assistant_message.id,
        current_question=payload.content,
        current_answer=answer,
        conversation_history=[{"role": item.role, "content": item.content} for item in history],
        constraints=[{"type": item.constraint_type, "content": item.content} for item in constraints],
    )
    analysis = AnalysisResult(session_id=session_id, message_id=assistant_message.id, **(await analyze(analysis_payload)))
    db.add(analysis)
    db.commit()
    return ChatResponse(user_message=user_message, assistant_message=assistant_message)
