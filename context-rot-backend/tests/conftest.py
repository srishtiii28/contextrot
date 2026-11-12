"""Tests use a disposable database and never touch the demo database."""
import os
import tempfile
from pathlib import Path
import pytest

TEST_DIR=Path(tempfile.mkdtemp(prefix="context-rot-tests-"))
os.environ["DATABASE_URL"]="sqlite:///"+(TEST_DIR/"test.db").as_posix()
os.environ["LLM_PROVIDER"]="offline"
os.environ["EMBEDDED_WORKER"]="false"
os.environ["ENABLE_DEMO_FAULTS"]="true"
os.environ["RAG_SERVICE_URL"]=""
os.environ["ROT_SERVICE_URL"]=""


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:yield c


@pytest.fixture
def account(client):
    from uuid import uuid4
    name="test_"+uuid4().hex[:12]
    assert client.post("/auth/signup",json=dict(username=name,email=name+"@example.com",password="test-password-123")).status_code==201
    login=client.post("/auth/login",json=dict(username=name,password="test-password-123"))
    return {"Authorization":"Bearer "+login.json()["access_token"]}
