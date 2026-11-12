from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import Document, KnowledgeBase, User
from app.integrations.rag_client import RetrievalRequest, RetrievalResponse, retrieve
from app.services.rag_service import ingest_pdf
from app.routes.dependencies import current_user
from app.services.session_service import owned_session

router = APIRouter(prefix="/integrations", tags=["integration contracts"])


@router.post("/rag/retrieve", response_model=RetrievalResponse)
async def rag_retrieve(payload: RetrievalRequest, db: Session = Depends(get_db), user: User = Depends(current_user)) -> RetrievalResponse:
    owned_session(db, session_id=payload.session_id, user_id=user.id)
    if payload.knowledge_base_id is not None and db.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.id == payload.knowledge_base_id,
            KnowledgeBase.owner_id == user.id,
        )
    ) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found")
    return await retrieve(payload, db=db)


@router.post("/sessions/{session_id}/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(
    session_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> dict[str, int | str]:
    owned_session(db, session_id=session_id, user_id=user.id)
    filename = file.filename or "uploaded.pdf"
    if file.content_type not in {"application/pdf", "application/octet-stream"} and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only PDF files are supported.")
    try:
        document = ingest_pdf(db, session_id=session_id, filename=filename, content=await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"id": document.id, "filename": document.filename, "chunks": len(document.chunks)}


@router.get("/sessions/{session_id}/documents")
def list_documents(
    session_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> list[dict[str, int | str]]:
    owned_session(db, session_id=session_id, user_id=user.id)
    documents = list(
        db.scalars(
            select(Document)
            .where(Document.session_id == session_id)
            .order_by(Document.created_at.desc(), Document.id.desc())
        )
    )
    return [
        {
            "id": document.id,
            "filename": document.filename,
            "content_type": document.content_type,
            "chunks": len(document.chunks),
        }
        for document in documents
    ]
