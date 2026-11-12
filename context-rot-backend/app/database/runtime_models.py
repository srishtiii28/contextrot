"""Additive tables: existing users/sessions/messages remain intact."""
from datetime import datetime
from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from app.database.database import Base


class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    digest: Mapped[str] = mapped_column(String(64), unique=True)
    prefix: Mapped[str] = mapped_column(String(16))
    revoked: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class MessageState(Base):
    __tablename__ = "message_states"
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"), primary_key=True)
    status: Mapped[str] = mapped_column(String(30), default="pending")
    sources: Mapped[list] = mapped_column(JSON, default=list)
    correction_of: Mapped[int | None] = mapped_column(ForeignKey("messages.id"), nullable=True)
    generation_ms: Mapped[float] = mapped_column(Float, default=0)


class Job(Base):
    __tablename__ = "reliability_jobs"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"), unique=True)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    claim: Mapped[str | None] = mapped_column(String(40), nullable=True)
    lease_until: Mapped[float] = mapped_column(Float, default=0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Event(Base):
    __tablename__ = "session_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    kind: Mapped[str] = mapped_column(String(50))
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class CourseChunk(Base):
    __tablename__ = "course_chunks"
    id: Mapped[int] = mapped_column(primary_key=True)
    topic: Mapped[str] = mapped_column(String(120))
    source: Mapped[str] = mapped_column(String(200))
    page: Mapped[int] = mapped_column(Integer, default=1)
    text: Mapped[str] = mapped_column(Text)


class SessionLease(Base):
    __tablename__ = "session_submission_leases"
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), primary_key=True)
    token: Mapped[str | None] = mapped_column(String(40), nullable=True)
    expires_at: Mapped[float] = mapped_column(Float,default=0)
