from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    text: str
    source: str
    page: int | None = None
    score: float = 0.0
    topic: str = ""


class RetrievalRequest(BaseModel):
    query: str
    session_id: int
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalResponse(BaseModel):
    chunks: list[RetrievedChunk]
    in_scope: bool = True


async def retrieve(request: RetrievalRequest) -> RetrievalResponse:
    from app.core.config import get_settings
    from app.services.course_service import search
    import httpx
    settings = get_settings()
    if settings.rag_service_url:
        headers = {"X-API-Key": settings.integration_api_key} if settings.integration_api_key else {}
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(settings.rag_service_url, json=request.model_dump(), headers=headers)
            response.raise_for_status()
            result=RetrievalResponse.model_validate(response.json())
            if not result.in_scope:result.chunks=[]
            return result
    return RetrievalResponse(chunks=search(request.query, request.top_k))

