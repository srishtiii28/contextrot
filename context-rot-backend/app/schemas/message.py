from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=12000)


class MessageResponse(ORMModel):
    id: int
    role: str
    content: str
    agent_type: str | None
    created_at: datetime


class ChatResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse

