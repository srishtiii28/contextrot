from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app


def test_member_one_mvp_flow() -> None:
    suffix = uuid4().hex[:8]
    with TestClient(app) as client:
        signup = client.post("/auth/signup", json={"username": f"student_{suffix}", "email": f"student_{suffix}@example.com", "password": "securepass123"})
        assert signup.status_code == 201
        login = client.post("/auth/login", json={"username": f"student_{suffix}", "password": "securepass123"})
        assert login.status_code == 200
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        created = client.post("/sessions", json={"title": "Entropy revision", "subject": "Physics"}, headers=headers)
        assert created.status_code == 201
        session_id = created.json()["id"]
        chat = client.post(f"/sessions/{session_id}/messages", json={"content": "Explain entropy."}, headers=headers)
        assert chat.status_code == 201
        assert chat.json()["assistant_message"]["role"] == "assistant"
        history = client.get(f"/sessions/{session_id}/history", headers=headers)
        assert len(history.json()) == 2
        health = client.get(f"/sessions/{session_id}/health", headers=headers)
        assert health.json()["current_score"] == 100.0
