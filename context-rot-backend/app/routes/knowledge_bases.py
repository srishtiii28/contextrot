from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import Document, KnowledgeBase, User
from app.routes.dependencies import current_user
from app.services.rag_service import ingest_pdf

router = APIRouter(prefix="/knowledge-bases", tags=["content management"])


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    subject: str | None = Field(default=None, max_length=120)


class KnowledgeBaseResponse(BaseModel):
    id: int
    owner_id: int
    name: str
    subject: str | None

    model_config = {"from_attributes": True}


@router.post("", response_model=KnowledgeBaseResponse, status_code=status.HTTP_201_CREATED)
def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> KnowledgeBase:
    knowledge_base = KnowledgeBase(owner_id=user.id, name=payload.name, subject=payload.subject)
    db.add(knowledge_base)
    db.commit()
    db.refresh(knowledge_base)
    return knowledge_base


@router.get("", response_model=list[KnowledgeBaseResponse])
def list_knowledge_bases(
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> list[KnowledgeBase]:
    return list(
        db.scalars(
            select(KnowledgeBase)
            .order_by(KnowledgeBase.name.asc())
        )
    )


@router.post("/{knowledge_base_id}/documents", status_code=status.HTTP_201_CREATED)
async def upload_reference_document(
    knowledge_base_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> dict[str, int | str]:
    knowledge_base = db.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.owner_id == user.id,
        )
    )
    if knowledge_base is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found")
    filename = file.filename or "reference.pdf"
    if file.content_type not in {"application/pdf", "application/octet-stream"} and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only PDF files are supported.")
    try:
        document = ingest_pdf(
            db,
            knowledge_base_id=knowledge_base_id,
            filename=filename,
            content=await file.read(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"id": document.id, "filename": document.filename, "chunks": len(document.chunks)}


@router.get("/{knowledge_base_id}/documents")
def list_reference_documents(
    knowledge_base_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> list[dict[str, int | str]]:
    knowledge_base = db.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.owner_id == user.id,
        )
    )
    if knowledge_base is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found")
    documents = list(
        db.scalars(
            select(Document)
            .where(Document.knowledge_base_id == knowledge_base_id)
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
