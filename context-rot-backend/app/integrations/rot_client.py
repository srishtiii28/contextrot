"""Detector boundary. ROT_SERVICE_URL points at an external detector; otherwise the in-process
reliability layer runs (app/reliability, mode chosen by RELIABILITY_DETECTOR)."""
import httpx
from app.core.config import get_settings
from app.schemas.analysis import AnalysisInput, DetectorResult


async def analyze(payload: AnalysisInput) -> dict:
    settings = get_settings()
    if settings.rot_service_url:
        headers = {"X-API-Key": settings.integration_api_key} if settings.integration_api_key else {}
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(settings.rot_service_url, json=payload.model_dump(), headers=headers)
            response.raise_for_status()
            return DetectorResult.model_validate(response.json()).model_dump()
    from app import reliability
    return DetectorResult.model_validate(await reliability.score(payload)).model_dump()
