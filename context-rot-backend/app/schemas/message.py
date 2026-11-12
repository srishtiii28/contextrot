from datetime import datetime

from pydantic import BaseModel, Field
from typing import Literal

from app.schemas.common import ORMModel


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=12000)
    demo_fault: Literal["contradiction", "instruction", "drift", "persistent"] | None = None


class MessageResponse(ORMModel):
    id: int
    role: str
    content: str
    agent_type: str | None
    created_at: datetime
    status: str = "saved"
    sources: list[dict] = Field(default_factory=list)
    correction_of: int | None = None
    generation_ms: float = 0


class ChatResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse
    job_id: int | None = None

