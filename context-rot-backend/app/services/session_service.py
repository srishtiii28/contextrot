from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import StudySession


def owned_session(db: Session, *, session_id: int, user_id: int) -> StudySession:
    study_session = db.scalar(select(StudySession).where(StudySession.id == session_id, StudySession.user_id == user_id))
    if study_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return study_session

