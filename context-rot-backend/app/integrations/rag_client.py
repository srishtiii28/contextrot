from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    text: str
    source: str
    page: int | None = None
    score: float = 0.0


class RetrievalRequest(BaseModel):
    query: str
    session_id: int
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalResponse(BaseModel):
    chunks: list[RetrievedChunk]


async def retrieve(request: RetrievalRequest) -> RetrievalResponse:
    """Member 3 replaces this mock with its vector-retrieval implementation."""
    return RetrievalResponse(chunks=[])

