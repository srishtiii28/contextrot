from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.database import Base


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    sessions: Mapped[list["StudySession"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    knowledge_bases: Mapped[list["KnowledgeBase"]] = relationship(back_populates="owner", cascade="all, delete-orphan")


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    subject: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    owner: Mapped[User] = relationship(back_populates="knowledge_bases")
    sessions: Mapped[list["StudySession"]] = relationship(back_populates="knowledge_base")
    documents: Mapped[list["Document"]] = relationship(back_populates="knowledge_base", cascade="all, delete-orphan")


class StudySession(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    subject: Mapped[str | None] = mapped_column(String(120), nullable=True)
    knowledge_base_id: Mapped[int | None] = mapped_column(ForeignKey("knowledge_bases.id"), nullable=True, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    user: Mapped[User] = relationship(back_populates="sessions")
    messages: Mapped[list["Message"]] = relationship(back_populates="session", cascade="all, delete-orphan")
    constraints: Mapped[list["Constraint"]] = relationship(back_populates="session", cascade="all, delete-orphan")
    analyses: Mapped[list["AnalysisResult"]] = relationship(back_populates="session", cascade="all, delete-orphan")
    documents: Mapped[list["Document"]] = relationship(back_populates="session", cascade="all, delete-orphan")
    knowledge_base: Mapped[KnowledgeBase | None] = relationship(back_populates="sessions")


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    agent_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    image_data: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    session: Mapped[StudySession] = relationship(back_populates="messages")


class Constraint(Base):
    __tablename__ = "constraints"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    constraint_type: Mapped[str] = mapped_column(String(100))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    session: Mapped[StudySession] = relationship(back_populates="constraints")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    message_id: Mapped[int | None] = mapped_column(ForeignKey("messages.id"), nullable=True, index=True)
    topic_score: Mapped[float] = mapped_column(Float, default=1.0)
    contradiction_score: Mapped[float] = mapped_column(Float, default=0.0)
    memory_score: Mapped[float] = mapped_column(Float, default=1.0)
    scope_score: Mapped[float] = mapped_column(Float, default=1.0)
    health_score: Mapped[float] = mapped_column(Float)
    topic_drift: Mapped[bool] = mapped_column(Boolean, default=False)
    contradiction_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    memory_violation: Mapped[bool] = mapped_column(Boolean, default=False)
    alert: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    session: Mapped[StudySession] = relationship(back_populates="analyses")


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int | None] = mapped_column(ForeignKey("sessions.id"), nullable=True, index=True)
    knowledge_base_id: Mapped[int | None] = mapped_column(ForeignKey("knowledge_bases.id"), nullable=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    session: Mapped[StudySession] = relationship(back_populates="documents")
    knowledge_base: Mapped[KnowledgeBase | None] = relationship(back_populates="documents")
    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    images: Mapped[list["DocumentImage"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start_char: Mapped[int] = mapped_column(Integer)
    end_char: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    document: Mapped[Document] = relationship(back_populates="chunks")


class DocumentImage(Base):
    __tablename__ = "document_images"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    page: Mapped[int] = mapped_column(Integer)
    image_filename: Mapped[str] = mapped_column(String(255))
    caption: Mapped[str] = mapped_column(Text, default="")
    image_data: Mapped[bytes] = mapped_column(LargeBinary)
    document: Mapped[Document] = relationship(back_populates="images")
