"""Replaceable detector boundary. Local rules are an unvalidated demo baseline."""
import re
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
    answer = payload.current_answer.lower()
    contradiction = bool(re.search(r"primary key (?:can|may|allows?) (?:contain |have )?(?:null|duplicate)", answer))
    drift = "photosynthesis" in answer and "photosynthesis" not in payload.current_question.lower()
    violations = []
    for constraint in payload.constraints:
        text = constraint["content"].lower()
        match = re.search(r"(?:at most|max(?:imum)?|no more than)\s+(\d+)\s+sentences?", text)
        prose = re.sub(r"\[[^\]]*\]", "", payload.current_answer)
        if match and len([s for s in re.split(r"[.!?]+(?:\s|$)", prose) if s.strip()]) > int(match[1]):
            violations.append("sentence limit")
        if "bullet" in text and not re.search(r"(?m)^\s*[-*] ", payload.current_answer):
            violations.append("bullet format")
    alerts = []
    if contradiction: alerts.append("Claim conflicts with the course rule for primary keys.")
    if drift: alerts.append("Answer contains an unrelated topic.")
    if violations: alerts.append("Ignored instruction: " + ", ".join(violations) + ".")
    return DetectorResult(topic_score=0 if drift else 1, contradiction_score=float(contradiction), memory_score=0 if violations else 1,
        scope_score=0 if drift else 1, health_score=max(0,100-70*contradiction-55*bool(violations)-70*drift),
        topic_drift=drift, contradiction_detected=contradiction, memory_violation=bool(violations), alert=" ".join(alerts) or None).model_dump()

