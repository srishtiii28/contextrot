from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


def _engine_url() -> str:
    return get_settings().database_url


engine = create_engine(
    _engine_url(),
    connect_args={"check_same_thread": False} if _engine_url().startswith("sqlite") else {},
    **({"poolclass":NullPool} if _engine_url().startswith("sqlite") else {}),
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

if _engine_url().startswith("sqlite"):
    @event.listens_for(engine,"connect")
    def sqlite_pragmas(connection,_):
        cursor=connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=10000")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


def get_db():
    database = SessionLocal()
    try:
        yield database
    finally:
        database.close()


def create_tables() -> None:
    from app.database import models  # noqa: F401
    from app.database import runtime_models  # noqa: F401

    Base.metadata.create_all(bind=engine)

