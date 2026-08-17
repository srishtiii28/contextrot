from fastapi import APIRouter, Depends

from app.database.models import User
from app.integrations.rag_client import RetrievalRequest, RetrievalResponse, retrieve
from app.routes.dependencies import current_user

router = APIRouter(prefix="/integrations", tags=["integration contracts"])


@router.post("/rag/retrieve", response_model=RetrievalResponse)
async def rag_retrieve(payload: RetrievalRequest, user: User = Depends(current_user)) -> RetrievalResponse:
    return await retrieve(payload)

