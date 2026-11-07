from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.services.rag_service import cosine_similarity, deserialize_embedding, embed, retrieve_chunks


class RetrievedChunk(BaseModel):
    text: str
    source: str
    page: int | None = None
    score: float = 0.0
    document_id: int | None = None


class RetrievalRequest(BaseModel):
    query: str
    session_id: int
    top_k: int = Field(default=3, ge=1, le=20)
    knowledge_base_id: int | None = Field(default=None, ge=1)


class RetrievalResponse(BaseModel):
    chunks: list[RetrievedChunk]


async def retrieve(request: RetrievalRequest, db: Session) -> RetrievalResponse:
    chunks = retrieve_chunks(
        db,
        session_id=request.session_id,
        knowledge_base_id=request.knowledge_base_id,
        query=request.query,
        top_k=request.top_k,
    )
    query_embedding = embed(request.query)
    return RetrievalResponse(
        chunks=[
            RetrievedChunk(
                text=chunk.text,
                source=chunk.document.filename,
                page=chunk.page,
                score=cosine_similarity(query_embedding, deserialize_embedding(chunk.embedding)),
                document_id=chunk.document_id,
            )
            for chunk in chunks
        ]
    )
