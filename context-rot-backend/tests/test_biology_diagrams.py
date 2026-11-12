import time
from uuid import uuid4
from fastapi.testclient import TestClient
from app.main import app
from app.database.database import SessionLocal
from app.database.models import Document, DocumentImage
from app.services.rag_service import retrieve_diagram


def test_biology_conversational_diagram_retrieval_unit():
    """Unit test for retrieve_diagram with conversational follow-up queries and history."""
    db = SessionLocal()
    try:
        # 1. Turn 1: Explain Dicot stem, Turn 2: "diagram of the above"
        history_dicot = [
            {"role": "user", "content": "Explain the transverse section of a dicot stem."},
            {"role": "assistant", "content": "A dicot stem has an outer epidermis, hypodermis, cortex, endodermis, pericycle, vascular bundles arranged in a ring, and pith."},
        ]
        img_dicot = retrieve_diagram(
            db,
            session_id=-999,
            query="diagram of the above",
            knowledge_base_id=1,
            conversation_history=history_dicot,
        )
        assert img_dicot is not None
        assert "Figure 6.4" in img_dicot.caption
        assert "(a) Dicot" in img_dicot.caption and "(b) Monocot" in img_dicot.caption
        assert img_dicot.page == 87
        # Ensure it does NOT match Adrenal gland or any unrelated figure
        assert "adrenal" not in img_dicot.caption.lower()

        # 2. Turn 1: Explain Monocot root, Turn 2: "diagram of the above"
        history_root = [
            {"role": "user", "content": "What is the internal structure and anatomy of monocot root?"},
            {"role": "assistant", "content": "A monocot root has epiblema, cortex, endodermis, pericycle, polyarch xylem bundles, and a large pith."},
        ]
        img_root = retrieve_diagram(
            db,
            session_id=-999,
            query="diagram of the above",
            knowledge_base_id=1,
            conversation_history=history_root,
        )
        assert img_root is not None
        assert "Figure 6.3" in img_root.caption
        assert "(a) Dicot root" in img_root.caption and "(b) Monocot root" in img_root.caption
        assert img_root.page == 86

        # 3. Turn 3: "explain all its labellings"
        history_labellings = [
            *history_dicot,
            {"role": "user", "content": "diagram of the above"},
            {"role": "assistant", "content": "As shown in the diagram above..."},
        ]
        img_labels = retrieve_diagram(
            db,
            session_id=-999,
            query="explain all its labellings",
            knowledge_base_id=1,
            conversation_history=history_labellings,
        )
        assert img_labels is not None
        assert "Figure 6.4" in img_labels.caption

        # 4. Grouped diagram query for either sub-part returns the complete plate
        img_monocot_stem = retrieve_diagram(db, session_id=-999, query="diagram of monocot stem", knowledge_base_id=1)
        assert img_monocot_stem is not None
        assert img_monocot_stem.id == img_dicot.id

        img_dicot_root = retrieve_diagram(db, session_id=-999, query="diagram of dicot root", knowledge_base_id=1)
        assert img_dicot_root is not None
        assert img_dicot_root.id == img_root.id

        # 5. Referential word "above" without history must NOT return Adrenal gland
        img_orphan = retrieve_diagram(db, session_id=-999, query="diagram of the above", knowledge_base_id=1, conversation_history=[])
        assert img_orphan is None
    finally:
        db.close()


def test_biology_conversational_chat_flow():
    """End-to-end integration test of user asking about Dicot stem then 'diagram of the above'."""
    suffix = uuid4().hex[:8]
    with TestClient(app) as client:
        signup = client.post(
            "/auth/signup",
            json={
                "username": f"bio_diag_{suffix}",
                "email": f"bio_diag_{suffix}@example.com",
                "password": "securepass123",
            },
        )
        assert signup.status_code == 201
        login = client.post("/auth/login", json={"username": f"bio_diag_{suffix}", "password": "securepass123"})
        headers = {"Authorization": "Bearer " + login.json()["access_token"]}

        # Create session linked to Biology library (ID: 1)
        session_res = client.post(
            "/sessions",
            json={"title": "Plant Anatomy Study", "knowledge_base_id": 1},
            headers=headers,
        )
        assert session_res.status_code == 201
        session_id = session_res.json()["id"]

        # Turn 1: User asks about dicot stem
        turn1_res = client.post(
            f"/sessions/{session_id}/messages",
            json={"content": "Explain the anatomy and internal structure of a dicot stem."},
            headers=headers,
        )
        assert turn1_res.status_code == 201
        t1_body = turn1_res.json()
        assert t1_body["assistant_message"]["role"] == "assistant"
        t1_ans = t1_body["assistant_message"]["content"]
        assert any(k in t1_ans.lower() for k in ("epidermis", "cortex", "vascular", "xylem", "phloem", "pericycle"))

        import asyncio
        from app.services.worker import process_one
        while asyncio.run(process_one()):
            pass

        # Turn 2: User asks "diagram of the above"
        time.sleep(2)
        turn2_res = client.post(
            f"/sessions/{session_id}/messages",
            json={"content": "diagram of the above"},
            headers=headers,
        )
        assert turn2_res.status_code == 201
        t2_body = turn2_res.json()
        t2_ans = t2_body["assistant_message"]["content"]

        # MUST contain the diagram Markdown image tag referencing Figure 6.4
        assert "/images/" in t2_ans
        assert "Figure 6.4" in t2_ans or "T.S. of stem" in t2_ans

        # MUST NOT say "diagram is not available in the uploaded notes"
        assert "not available in the uploaded notes" not in t2_ans.lower()
        assert "not available in the notes" not in t2_ans.lower()

        # MUST explain the anatomical labellings and layers
        assert any(layer in t2_ans.lower() for layer in ("epidermis", "cortex", "hypodermis", "endodermis", "pericycle", "vascular bundle", "xylem", "phloem", "pith"))

        # MUST NOT show Adrenal gland or an unrelated diagram
        assert "adrenal" not in t2_ans.lower()
