import pytest
from uuid import uuid4
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.main import app
from app.database.database import SessionLocal
from app.database.models import KnowledgeBase, StudySession
from app.services.llm_service import (
    normalize_subject_category,
    _build_subject_system_prompt,
    _response_guidance,
)


def test_normalize_subject_category():
    assert normalize_subject_category("Maths") == "mathematics"
    assert normalize_subject_category("Mathematics") == "mathematics"
    assert normalize_subject_category("Algebra & Geometry") == "mathematics"
    assert normalize_subject_category("Biology") == "biology"
    assert normalize_subject_category("Bio notes") == "biology"
    assert normalize_subject_category("Physics") == "physics"
    assert normalize_subject_category("Grade 12 Physics") == "physics"
    assert normalize_subject_category("Chemistry") == "chemistry"
    assert normalize_subject_category("Organic Chemistry") == "chemistry"

    # Fallback to context text
    assert normalize_subject_category(None, "Solving linear equations in two variables") == "mathematics"
    assert normalize_subject_category("", "Dicot stem and monocot root anatomy") == "biology"
    assert normalize_subject_category(None, "Electric charge and Coulomb force") == "physics"
    assert normalize_subject_category(None, "Balanced chemical equation and stoichiometry") == "chemistry"
    assert normalize_subject_category(None, "random general text") == "general"


def test_build_subject_system_prompt():
    math_prompt = _build_subject_system_prompt("mathematics")
    assert "MATHEMATICS & GEOMETRY SPECIFIC REQUIREMENTS" in math_prompt
    assert "COINCIDENT LINES" in math_prompt
    assert "ARITHMETIC SANITY CHECK" in math_prompt
    assert "LATEX MATH NOTATION" in math_prompt
    assert "PROPER LEGIBLE TABLES" in math_prompt
    assert "ACCURATE 2D GRAPHS & GEOMETRY" in math_prompt

    bio_prompt = _build_subject_system_prompt("biology")
    assert "BIOLOGY SPECIFIC REQUIREMENTS" in bio_prompt
    assert "DIAGRAM & FIGURE FIDELITY" in bio_prompt
    assert "EXAM-READY PEDAGOGICAL PROSE" in bio_prompt
    assert "ACCURATE MCQs" in bio_prompt

    phys_prompt = _build_subject_system_prompt("physics")
    assert "PHYSICS SPECIFIC REQUIREMENTS" in phys_prompt
    assert "SI units" in phys_prompt

    chem_prompt = _build_subject_system_prompt("chemistry")
    assert "CHEMISTRY SPECIFIC REQUIREMENTS" in chem_prompt
    assert "CHEMICAL REACTIONS & EQUATIONS" in chem_prompt


def test_subject_response_guidance():
    math_guidance = _response_guidance("solve Roman's purchase graphically", subject_category="mathematics")
    assert "coordinate table" in math_guidance.lower()
    assert "points" in math_guidance.lower()

    bio_guidance = _response_guidance("draw diagram of dicot stem", subject_category="biology")
    assert "biological structure" in bio_guidance.lower()
    assert "textbook diagram" in bio_guidance.lower()


def test_session_creation_inherits_library_subject():
    suffix = uuid4().hex[:8]
    with TestClient(app) as client:
        # 1. Signup and login
        signup = client.post(
            "/auth/signup",
            json={
                "username": f"learner_{suffix}",
                "email": f"learner_{suffix}@example.com",
                "password": "securepass123",
            },
        )
        assert signup.status_code == 201
        login = client.post("/auth/login", json={"username": f"learner_{suffix}", "password": "securepass123"})
        assert login.status_code == 200
        headers = {"Authorization": "Bearer " + login.json()["access_token"]}

        # 2. List libraries - should see all 4 core libraries
        kb_resp = client.get("/knowledge-bases", headers=headers)
        assert kb_resp.status_code == 200
        kbs = kb_resp.json()
        kb_names = {k["name"] for k in kbs}
        assert "Biology" in kb_names
        assert "Maths" in kb_names
        assert "Physics" in kb_names
        assert "Chemistry" in kb_names

        # 3. Create session with Maths library
        math_kb = next(k for k in kbs if k["name"] == "Maths")
        math_sess_resp = client.post(
            "/sessions",
            json={
                "title": "Algebra Revision",
                "knowledge_base_id": math_kb["id"],
                # subject omitted to test auto-inheritance
            },
            headers=headers,
        )
        assert math_sess_resp.status_code == 201
        math_sess = math_sess_resp.json()
        assert math_sess["subject"] == "Mathematics"

        # 4. Create session with Biology library
        bio_kb = next(k for k in kbs if k["name"] == "Biology")
        bio_sess_resp = client.post(
            "/sessions",
            json={
                "title": "Botany Revision",
                "knowledge_base_id": bio_kb["id"],
            },
            headers=headers,
        )
        assert bio_sess_resp.status_code == 201
        bio_sess = bio_sess_resp.json()
        assert bio_sess["subject"] == "Biology"
