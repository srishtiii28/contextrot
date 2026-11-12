from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.services.session_service import owned_session

from app.database.models import User
from app.integrations.rag_client import RetrievalRequest, RetrievalResponse, retrieve
from app.routes.dependencies import current_user

router = APIRouter(prefix="/integrations", tags=["integration contracts"])


@router.post("/rag/retrieve", response_model=RetrievalResponse)
async def rag_retrieve(payload: RetrievalRequest, user: User = Depends(current_user), db: Session = Depends(get_db)) -> RetrievalResponse:
    owned_session(db,session_id=payload.session_id,user_id=user.id)
    return await retrieve(payload)

