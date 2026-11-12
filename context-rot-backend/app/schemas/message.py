from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import ORMModel


class MessageCreate(BaseModel):
    content: str = Field(default="", max_length=12000)
    image_data: str | None = None
    demo_fault: Literal["contradiction", "instruction", "drift", "persistent"] | None = None

    @model_validator(mode="after")
    def validate_content_or_image(self):
        if not (self.content or "").strip() and not self.image_data:
            raise ValueError("Either message content or an attached image must be provided.")
        return self


class MessageResponse(ORMModel):
    id: int
    role: str
    content: str
    agent_type: str | None = None
    created_at: datetime
    status: str = "saved"
    sources: list[dict] = Field(default_factory=list)
    correction_of: int | None = None
    generation_ms: float = 0
    image_data: str | None = None


class AnswerSource(BaseModel):
    source: str
    page: int | None = None
    score: float


class DiagramSource(BaseModel):
    id: int
    caption: str
    page: int
    url: str


class ChatResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse
    job_id: int | None = None
    sources: list[AnswerSource] = Field(default_factory=list)
    diagrams: list[DiagramSource] = Field(default_factory=list)
