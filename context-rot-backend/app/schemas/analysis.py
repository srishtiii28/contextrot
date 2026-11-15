from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class AnalysisInput(BaseModel):
    session_id: int
    message_id: int | None = None
    current_question: str
    current_answer: str
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    constraints: list[dict[str, str]] = Field(default_factory=list)
    retrieved_context: list[dict] = Field(default_factory=list)


FlagType = Literal["contradiction", "ignored_instruction", "course_conflict", "topic_drift"]


class Flag(BaseModel):
    """One piece of evidence behind a health score. Severity is a 0-1 detector signal, not a calibrated probability."""
    type: FlagType
    severity: float = Field(ge=0, le=1)
    claim: str | None = None
    evidence: str | None = None
    turn: int | None = None


class DetectorResult(BaseModel):
    topic_score: float = Field(ge=0, le=1)
    contradiction_score: float = Field(ge=0, le=1)
    memory_score: float = Field(ge=0, le=1)
    scope_score: float = Field(ge=0, le=1)
    health_score: float = Field(ge=0, le=100)
    topic_drift: bool
    contradiction_detected: bool
    memory_violation: bool
    alert: str | None = None
    flags: list[Flag] = Field(default_factory=list)
    detector: str | None = None
    latency_ms: float | None = None


class AnalysisResponse(ORMModel):
    id: int
    session_id: int
    message_id: int | None
    topic_score: float
    contradiction_score: float
    memory_score: float
    scope_score: float
    health_score: float
    topic_drift: bool
    contradiction_detected: bool
    memory_violation: bool
    alert: str | None
    flags: list[Flag] | None = None
    detector: str | None = None
    latency_ms: float | None = None
    created_at: datetime


class HealthResponse(BaseModel):
    session_id: int
    current_score: float | None
    trend: str
    alerts: list[str]

