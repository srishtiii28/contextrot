from app.schemas.analysis import AnalysisInput


async def analyze(payload: AnalysisInput) -> dict:
    """Member 2 replaces this deterministic placeholder with the rot engine."""
    return {
        "topic_score": 1.0,
        "contradiction_score": 0.0,
        "memory_score": 1.0,
        "scope_score": 1.0,
        "health_score": 100.0,
        "topic_drift": False,
        "contradiction_detected": False,
        "memory_violation": False,
        "alert": None,
    }

