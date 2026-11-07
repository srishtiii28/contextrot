from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import ORMModel


class MessageCreate(BaseModel):
    content: str = Field(default="", max_length=12000)
    image_data: str | None = None

    @model_validator(mode="after")
    def validate_content_or_image(self):
        if not (self.content or "").strip() and not self.image_data:
            raise ValueError("Either message content or an attached image must be provided.")
        return self


class MessageResponse(ORMModel):
    id: int
    role: str
    content: str
    agent_type: str | None
    created_at: datetime
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
    sources: list[AnswerSource] = Field(default_factory=list)
    diagrams: list[DiagramSource] = Field(default_factory=list)
