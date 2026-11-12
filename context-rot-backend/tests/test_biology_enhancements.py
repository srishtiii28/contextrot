import pytest
from app.database.database import SessionLocal
from app.services.rag_service import (
    retrieve_diagram,
    resolve_conversational_query,
    retrieve_chunks,
    check_scope_boundary,
)
from app.services.llm_service import _clean_chunk_content, _response_guidance


def test_cymose_inflorescence_not_contaminated_by_heart_history():
    """Verify that asking for cymose inflorescence when heart was in history matches Figure 5.8, NOT heart."""
    db = SessionLocal()
    try:
        history_heart = [
            {"role": "user", "content": "Explain the human heart, its chambers, and cardiac cycle."},
            {"role": "assistant", "content": "The human heart is a muscular organ with four chambers..."},
        ]
        q_cymose = "diagram of cymose inflorescence"
        resolved_msg, retrieval_q = resolve_conversational_query(q_cymose, history_heart)
        assert "heart" not in retrieval_q.lower()

        diag_cymose = retrieve_diagram(
            db,
            session_id=-999,
            query=q_cymose,
            knowledge_base_id=1,
            conversation_history=history_heart,
        )
        assert diag_cymose is not None
        assert "Figure 5.8" in diag_cymose.caption and "Cymose" in diag_cymose.caption
        assert "heart" not in diag_cymose.caption.lower()
    finally:
        db.close()


def test_referential_diagram_query_resolves_from_history():
    """Verify that 'diagram of the above' still correctly resolves to the active topic from history."""
    db = SessionLocal()
    try:
        history_dicot = [
            {"role": "user", "content": "Explain the transverse section of a dicot stem."},
            {"role": "assistant", "content": "A dicot stem has epidermis, cortex, vascular bundles..."},
        ]
        q_ref = "diagram of the above"
        resolved_ref, retrieval_ref = resolve_conversational_query(q_ref, history_dicot)
        assert "dicot stem" in retrieval_ref.lower()

        diag_ref = retrieve_diagram(
            db,
            session_id=-999,
            query=q_ref,
            knowledge_base_id=1,
            conversation_history=history_dicot,
        )
        assert diag_ref is not None
        assert "Figure 6.4" in diag_ref.caption
    finally:
        db.close()


def test_small_details_in_scope_and_relevance_ranking():
    """Verify that niche textbook details are in-scope and appear in top chunks."""
    db = SessionLocal()
    try:
        details = [
            ("What is phyllotaxy?", "phyllotaxy"),
            ("What is the aleurone layer?", "aleurone"),
            ("What are bulliform cells?", "bulliform"),
            ("What is parthenocarpy?", "parthenocarpy"),
            ("What is a perigynous flower?", "perigynous"),
            ("What is the casparian strip?", "casparian"),
            ("What is placentation?", "placentation"),
            ("What is aestivation?", "aestivation"),
        ]
        for query, keyword in details:
            chunks = retrieve_chunks(db, session_id=-999, query=query, top_k=6, knowledge_base_id=1)
            in_scope = check_scope_boundary(query, chunks, has_uploaded_docs=True)
            assert in_scope, f"{query} was incorrectly rejected as out-of-scope!"
            has_target = any(keyword in c.text.lower() for c in chunks[:3])
            assert has_target, f"{keyword} was missing from top-3 chunks for {query}!"
    finally:
        db.close()


def test_response_guidance_all_marks():
    """Verify response guidance handles 1, 2, 3, 4, 5, 6 marks cleanly."""
    g1 = _response_guidance("What is phyllotaxy? 1 mark")
    assert "30-50" in g1
    g2 = _response_guidance("What is phyllotaxy? 2 mark")
    assert "70-100" in g2
    g3 = _response_guidance("What is phyllotaxy? 3 mark")
    assert "110-150" in g3
    g4 = _response_guidance("What is phyllotaxy? 4 mark")
    assert "140-200" in g4
    g5 = _response_guidance("What is phyllotaxy? 5 mark")
    assert "180-260" in g5
    g6 = _response_guidance("What is phyllotaxy? 6 mark")
    assert "220-320" in g6


def test_clean_chunk_content():
    """Verify running textbook headers and footers are stripped."""
    dirty = "MORPHOLOGY OF FLOWERING PLANTS 67 The arrangement of leaves. 66 BIOLOGY Rationalised 2023-24"
    cleaned = _clean_chunk_content(dirty)
    assert "MORPHOLOGY OF FLOWERING PLANTS 67" not in cleaned
    assert "66 BIOLOGY" not in cleaned
    assert "Rationalised 2023-24" not in cleaned
    assert "The arrangement of leaves." in cleaned


def test_end_to_end_cymose_after_heart_chat_flow():
    """Integration test verifying asking cymose inflorescence after heart query shows Figure 5.8 at the top."""
    import time
    from uuid import uuid4
    from fastapi.testclient import TestClient
    from app.main import app

    suffix = uuid4().hex[:8]
    with TestClient(app) as client:
        signup = client.post(
            "/auth/signup",
            json={
                "username": f"cymose_usr_{suffix}",
                "email": f"cymose_{suffix}@example.com",
                "password": "securepass123",
            },
        )
        assert signup.status_code == 201
        login = client.post("/auth/login", json={"username": f"cymose_usr_{suffix}", "password": "securepass123"})
        headers = {"Authorization": "Bearer " + login.json()["access_token"]}

        # Create session linked to Biology library (ID: 1)
        session_res = client.post(
            "/sessions",
            json={"title": "Biology Study Session", "knowledge_base_id": 1},
            headers=headers,
        )
        assert session_res.status_code == 201
        session_id = session_res.json()["id"]

        # Turn 1: User asks about human heart
        turn1 = client.post(
            f"/sessions/{session_id}/messages",
            json={"content": "Explain the structure of the human heart."},
            headers=headers,
        )
        assert turn1.status_code == 201
        t1_ans = turn1.json()["assistant_message"]["content"]
        assert any(k in t1_ans.lower() for k in ("heart", "atria", "ventricle", "chamber"))

        import asyncio
        from app.services.worker import process_one
        while asyncio.run(process_one()):
            pass

        # Turn 2: User asks for diagram of cymose inflorescence
        time.sleep(1)
        turn2 = client.post(
            f"/sessions/{session_id}/messages",
            json={"content": "diagram of cymose inflorescence"},
            headers=headers,
        )
        assert turn2.status_code == 201
        t2_json = turn2.json()
        t2_ans = t2_json["assistant_message"]["content"]

        # MUST show Figure 5.8 Cymose inflorescence
        assert len(t2_json["diagrams"]) == 1
        assert "Figure 5.8" in t2_json["diagrams"][0]["caption"]
        assert "cymose" in t2_json["diagrams"][0]["caption"].lower()

        # MUST NOT show heart diagram
        assert "heart" not in t2_json["diagrams"][0]["caption"].lower()
        assert "adrenal" not in t2_json["diagrams"][0]["caption"].lower()

        # Diagram image MUST be at the top of the answer
        assert t2_ans.strip().startswith("![Figure 5.8")

