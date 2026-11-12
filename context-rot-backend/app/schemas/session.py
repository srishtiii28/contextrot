from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class SessionCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    subject: str | None = Field(default=None, max_length=120)
    knowledge_base_id: int | None = Field(default=None, ge=1)
    constraints: list[str] = Field(default_factory=list, max_length=20)


class SessionResponse(ORMModel):
    id: int
    title: str
    subject: str | None
    started_at: datetime
    ended_at: datetime | None
    knowledge_base_id: int | None = None
