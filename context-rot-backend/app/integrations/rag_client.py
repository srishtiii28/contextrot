from pydantic import BaseModel, Field
from sqlalchemy.orm import Session


class RetrievedChunk(BaseModel):
    text: str
    source: str
    page: int | None = None
    score: float = 0.0
    topic: str = ""
    document_id: int | None = None


class RetrievalRequest(BaseModel):
    query: str
    session_id: int
    top_k: int = Field(default=5, ge=1, le=20)
    knowledge_base_id: int | None = Field(default=None, ge=1)


class RetrievalResponse(BaseModel):
    chunks: list[RetrievedChunk]
    in_scope: bool = True


async def retrieve(request: RetrievalRequest, db: Session | None = None) -> RetrievalResponse:
    from app.core.config import get_settings
    settings = get_settings()

    if settings.rag_service_url:
        import httpx
        headers = {"X-API-Key": settings.integration_api_key} if settings.integration_api_key else {}
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(settings.rag_service_url, json=request.model_dump(), headers=headers)
            response.raise_for_status()
            result = RetrievalResponse.model_validate(response.json())
            if not result.in_scope:
                result.chunks = []
            return result

    # Local retrieval resolution:
    from app.database.database import SessionLocal
    from app.services.course_service import search
    from app.services.rag_service import (
        check_scope_boundary,
        cosine_similarity,
        deserialize_embedding,
        embed,
        retrieve_chunks,
    )
    from app.database.models import Document

    session_created = False
    if db is None:
        db = SessionLocal()
        session_created = True

    try:
        use_kb = False
        if request.knowledge_base_id is not None:
            use_kb = True
        else:
            has_session_docs = db.query(Document.id).filter(Document.session_id == request.session_id).first() is not None
            if has_session_docs:
                use_kb = True

        if use_kb:
            chunks = retrieve_chunks(
                db,
                session_id=request.session_id,
                knowledge_base_id=request.knowledge_base_id,
                query=request.query,
                top_k=request.top_k,
            )
            in_scope = check_scope_boundary(request.query, chunks, has_uploaded_docs=True)
            if not in_scope:
                return RetrievalResponse(chunks=[], in_scope=False)

            query_embedding = embed(request.query)
            retrieved_chunks = [
                RetrievedChunk(
                    text=chunk.text,
                    source=chunk.document.filename if chunk.document else "uploaded_document.pdf",
                    page=chunk.page,
                    score=cosine_similarity(query_embedding, deserialize_embedding(chunk.embedding)) if chunk.embedding else 0.0,
                    document_id=chunk.document_id,
                    topic="",
                )
                for chunk in chunks
            ]
            return RetrievalResponse(chunks=retrieved_chunks, in_scope=True)
        else:
            # Fallback to authored course notes (DBMS revision)
            fallback_matches = search(request.query, request.top_k)
            retrieved_chunks = [RetrievedChunk(**m) for m in fallback_matches]
            return RetrievalResponse(chunks=retrieved_chunks, in_scope=bool(retrieved_chunks))
    finally:
        if session_created:
            db.close()
