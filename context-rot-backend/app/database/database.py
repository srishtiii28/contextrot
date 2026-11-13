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
    connect_args={"check_same_thread": False, "timeout": 30} if _engine_url().startswith("sqlite") else {},
    **({"poolclass": NullPool} if _engine_url().startswith("sqlite") else {}),
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

if _engine_url().startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def sqlite_pragmas(connection, _):
        cursor = connection.cursor()
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
    if engine.url.get_backend_name() == "sqlite":
        with engine.begin() as connection:
            columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(sessions)")}
            if "knowledge_base_id" not in columns:
                connection.exec_driver_sql("ALTER TABLE sessions ADD COLUMN knowledge_base_id INTEGER")
            columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(documents)")}
            if "knowledge_base_id" not in columns:
                connection.exec_driver_sql("ALTER TABLE documents ADD COLUMN knowledge_base_id INTEGER")
            columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(messages)")}
            if "image_data" not in columns:
                connection.exec_driver_sql("ALTER TABLE messages ADD COLUMN image_data TEXT")
            columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(message_states)")}
            if "scope_warning" not in columns:
                connection.exec_driver_sql("ALTER TABLE message_states ADD COLUMN scope_warning TEXT")
