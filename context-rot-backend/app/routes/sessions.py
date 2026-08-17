from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import Message, StudySession, User
from app.routes.dependencies import current_user
from app.schemas.message import MessageResponse
from app.schemas.session import SessionCreate, SessionResponse
from app.services.session_service import owned_session

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(payload: SessionCreate, db: Session = Depends(get_db), user: User = Depends(current_user)) -> StudySession:
    study_session = StudySession(user_id=user.id, title=payload.title, subject=payload.subject)
    db.add(study_session)
    db.commit()
    db.refresh(study_session)
    return study_session


@router.get("", response_model=list[SessionResponse])
def list_sessions(db: Session = Depends(get_db), user: User = Depends(current_user)) -> list[StudySession]:
    return list(db.scalars(select(StudySession).where(StudySession.user_id == user.id).order_by(StudySession.started_at.desc())))


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)) -> StudySession:
    return owned_session(db, session_id=session_id, user_id=user.id)


@router.get("/{session_id}/history", response_model=list[MessageResponse])
def history(session_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)) -> list[Message]:
    owned_session(db, session_id=session_id, user_id=user.id)
    return list(db.scalars(select(Message).where(Message.session_id == session_id).order_by(Message.created_at, Message.id)))

