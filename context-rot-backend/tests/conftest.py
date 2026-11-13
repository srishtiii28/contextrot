"""Tests use a disposable database and never touch the demo database."""
import os
import shutil
import tempfile
from pathlib import Path
import pytest

TEST_DIR = Path(tempfile.mkdtemp(prefix="context-rot-tests-"))
test_db_path = TEST_DIR / "test.db"

# If context_rot.db with pre-indexed textbooks exists, copy it to disposable test.db
dev_db = Path("context_rot.db")
if dev_db.exists() and dev_db.stat().st_size > 1000000:
    try:
        shutil.copyfile(dev_db, test_db_path)
    except Exception:
        pass

os.environ["DATABASE_URL"] = "sqlite:///" + test_db_path.as_posix()
os.environ["LLM_PROVIDER"] = "offline"
os.environ["EMBEDDED_WORKER"] = "false"
os.environ["ENABLE_DEMO_FAULTS"] = "true"
os.environ["RAG_SERVICE_URL"] = ""
os.environ["ROT_SERVICE_URL"] = ""


@pytest.fixture(autouse=True)
def setup_test_db():
    from app.database.database import create_tables
    from app.services.course_service import seed_course
    create_tables()
    seed_course()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def account(client):
    from uuid import uuid4
    name = "test_" + uuid4().hex[:12]
    assert client.post("/auth/signup", json=dict(username=name, email=name + "@example.com", password="test-password-123")).status_code == 201
    login = client.post("/auth/login", json=dict(username=name, password="test-password-123"))
    return {"Authorization": "Bearer " + login.json()["access_token"]}
