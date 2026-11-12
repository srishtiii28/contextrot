from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import AnalysisResult, Message, User
from app.integrations.rot_client import analyze
from app.routes.dependencies import current_user
from app.schemas.analysis import AnalysisInput, AnalysisResponse, HealthResponse
from app.services.session_service import owned_session

router = APIRouter(tags=["analysis"])


@router.post("/integrations/rot/analyze", response_model=AnalysisResponse)
async def submit_analysis(payload: AnalysisInput, db: Session = Depends(get_db), user: User = Depends(current_user)) -> AnalysisResult:
    owned_session(db, session_id=payload.session_id, user_id=user.id)
    if payload.message_id is not None:
        message=db.get(Message,payload.message_id)
        if not message or message.session_id!=payload.session_id:
            from fastapi import HTTPException
            raise HTTPException(404,"Message not found in this session")
    result = AnalysisResult(session_id=payload.session_id, message_id=payload.message_id, **(await analyze(payload)))
    db.add(result)
    db.commit()
    db.refresh(result)
    return result


@router.get("/sessions/{session_id}/analysis", response_model=list[AnalysisResponse])
def list_analysis(session_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)) -> list[AnalysisResult]:
    owned_session(db, session_id=session_id, user_id=user.id)
    return list(db.scalars(select(AnalysisResult).where(AnalysisResult.session_id == session_id).order_by(AnalysisResult.created_at, AnalysisResult.id)))


@router.get("/sessions/{session_id}/health", response_model=HealthResponse)
def health(session_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)) -> HealthResponse:
    owned_session(db, session_id=session_id, user_id=user.id)
    from app.database.runtime_models import MessageState
    latest_message=db.scalar(select(Message).where(Message.session_id==session_id,Message.role=="assistant").order_by(Message.id.desc()).limit(1))
    latest_state=db.get(MessageState,latest_message.id) if latest_message else None
    if latest_state and latest_state.status in {"pending","check_failed","refused"}:
        return HealthResponse(session_id=session_id,current_score=None,trend="unavailable",alerts=["Latest answer: "+latest_state.status])
    records = list(db.scalars(select(AnalysisResult).where(AnalysisResult.session_id == session_id).order_by(AnalysisResult.created_at.desc(), AnalysisResult.id.desc()).limit(2)))
    if not records:
        return HealthResponse(session_id=session_id, current_score=None, trend="unavailable", alerts=[])
    latest = records[0]
    trend = "stable" if len(records) == 1 else ("improving" if latest.health_score > records[1].health_score else "degrading" if latest.health_score < records[1].health_score else "stable")
    return HealthResponse(session_id=session_id, current_score=latest.health_score, trend=trend, alerts=[latest.alert] if latest.alert else [])

