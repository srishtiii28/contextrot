"""Reliability layer: scores each tutor answer against the session history, the student's instructions and
the retrieved course material. Output: health score H (0-100) and flags. See docs/RELIABILITY.md.

Detectors (RELIABILITY_DETECTOR):
  rules  offline demo baseline for the scripted faults only (default, no model needed)
  judge  v1, one LLM-judge call per answer (JUDGE_PROVIDER / JUDGE_MODEL)
  nli    v2, local NLI cross-encoder + MiniLM embeddings, no LLM call
"""
import asyncio
import time

from app.core.config import get_settings
from app.schemas.analysis import AnalysisInput

DETECTORS = ("rules", "judge", "nli")


async def _dispatch(name: str, payload: AnalysisInput) -> dict:
    if name == "rules":
        from app.reliability import rules
        return rules.score(payload)
    if name == "judge":
        from app.reliability import judge
        return await judge.score(payload)
    if name == "nli":
        from app.reliability import nli
        return await nli.score(payload)
    raise ValueError(f"RELIABILITY_DETECTOR must be one of {', '.join(DETECTORS)} (got {name!r})")


async def score(payload: AnalysisInput, detector: str | None = None) -> dict:
    """Errors propagate: the worker marks the answer `check_failed` instead of silently passing it."""
    settings = get_settings()
    name = (detector or settings.reliability_detector or "rules").lower()
    started = time.perf_counter()
    result = await asyncio.wait_for(_dispatch(name, payload), timeout=settings.reliability_timeout_seconds)
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    result["detector"] = result.get("detector") or name
    return result
