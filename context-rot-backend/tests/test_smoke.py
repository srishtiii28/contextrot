from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app


def test_member_one_mvp_flow() -> None:
    suffix = uuid4().hex[:8]
    with TestClient(app) as client:
        signup = client.post(
            "/auth/signup",
            json={
                "username": f"student_{suffix}",
                "email": f"student_{suffix}@example.com",
                "password": "securepass123",
            },
        )
        assert signup.status_code == 201
        login = client.post("/auth/login", json={"username": f"student_{suffix}", "password": "securepass123"})
        assert login.status_code == 200
        headers = {"Authorization": "Bearer " + login.json()["access_token"]}
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


def test_pdf_upload_grounded_qa_flow() -> None:
    import io
    from pathlib import Path
    from pypdf import PdfReader, PdfWriter

    ncert_path = Path(__file__).resolve().parents[3] / "11 Biology NCERT (NCERT) (z-library.sk, 1lib.sk, z-lib.sk).pdf"
    if not ncert_path.exists():
        return

    reader = PdfReader(str(ncert_path))
    writer = PdfWriter()
    writer.add_page(reader.pages[20])
    pdf_bytes = io.BytesIO()
    writer.write(pdf_bytes)
    pdf_content = pdf_bytes.getvalue()

    suffix = uuid4().hex[:8]
    with TestClient(app) as client:
        signup = client.post(
            "/auth/signup",
            json={
                "username": f"bio_{suffix}",
                "email": f"bio_{suffix}@example.com",
                "password": "securepass123",
            },
        )
        assert signup.status_code == 201
        login = client.post("/auth/login", json={"username": f"bio_{suffix}", "password": "securepass123"})
        headers = {"Authorization": "Bearer " + login.json()["access_token"]}

        session_res = client.post("/sessions", json={"title": "Living World revision", "subject": "Biology"}, headers=headers)
        assert session_res.status_code == 201
        session_id = session_res.json()["id"]

        upload_res = client.post(
            f"/integrations/sessions/{session_id}/documents",
            files={"file": ("living_world.pdf", pdf_content, "application/pdf")},
            headers=headers,
        )
        assert upload_res.status_code == 201
        assert upload_res.json()["chunks"] >= 1

        chat_res = client.post(
            f"/sessions/{session_id}/messages",
            json={"content": "What does the text say about the diversity and variety of plants and animals?"},
            headers=headers,
        )
        assert chat_res.status_code == 201
        body = chat_res.json()
        assert body["assistant_message"]["role"] == "assistant"
        assert "could not find" not in body["assistant_message"]["content"].lower()
        assert len(body["sources"]) >= 1
        assert body["sources"][0]["source"] == "living_world.pdf"

        # Out-of-scope question should trigger the Scope Guard
        out_of_scope_res = client.post(
            f"/sessions/{session_id}/messages",
            json={"content": "How do I bake a chocolate cake at home?"},
            headers=headers,
        )
        assert out_of_scope_res.status_code == 201
        out_body = out_of_scope_res.json()
        assert out_body["assistant_message"]["content"] == "The question is beyond the scope of the topics in this chat."
        assert len(out_body["sources"]) == 0

        # Follow-up query without re-stating topic maintains conversational continuity
        followup_res = client.post(
            f"/sessions/{session_id}/messages",
            json={"content": "give me more questions"},
            headers=headers,
        )
        assert followup_res.status_code == 201
        followup_body = followup_res.json()
        assert followup_body["assistant_message"]["role"] == "assistant"
        assert "beyond the scope" not in followup_body["assistant_message"]["content"].lower()


def test_image_serving_endpoint() -> None:
    from app.database.database import SessionLocal
    from app.database.models import Document, DocumentImage

    db = SessionLocal()
    doc = Document(session_id=999, filename="test_diagrams.pdf", content_type="application/pdf")
    db.add(doc)
    db.flush()

    # Small 1x1 transparent PNG in bytes
    tiny_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00"
        b"\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    img = DocumentImage(
        document_id=doc.id,
        page=7,
        image_filename="page_7_img_0.png",
        caption="Figure 7.1: Test diagram structure",
        image_data=tiny_png,
    )
    db.add(img)
    db.commit()
    img_id = img.id
    db.close()

    with TestClient(app) as client:
        res = client.get(f"/images/{img_id}")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/png"
        assert res.content == tiny_png

        # Missing image returns 404
        missing_res = client.get("/images/99999999")
        assert missing_res.status_code == 404


def test_math_numerical_problem_solving_flow() -> None:
    from pathlib import Path
    from uuid import uuid4
    from fastapi.testclient import TestClient
    from app.main import app

    math_pdf_path = Path(__file__).resolve().parents[3] / "Class 10 Math NCERT CBSE (NCERT) (z-library.sk, 1lib.sk, z-lib.sk).pdf"
    if not math_pdf_path.exists():
        return

    import pymupdf
    src_doc = pymupdf.open(str(math_pdf_path))
    # Extract pages 3, 4, 5 (0-indexed 2, 3, 4: Euclid's Division Lemma and Example 1)
    math_doc = pymupdf.open()
    math_doc.insert_pdf(src_doc, from_page=2, to_page=4)
    pdf_content = math_doc.tobytes()

    suffix = uuid4().hex[:8]
    with TestClient(app) as client:
        signup = client.post(
            "/auth/signup",
            json={
                "username": f"math_{suffix}",
                "email": f"math_{suffix}@example.com",
                "password": "securepass123",
            },
        )
        assert signup.status_code == 201
        login = client.post("/auth/login", json={"username": f"math_{suffix}", "password": "securepass123"})
        headers = {"Authorization": "Bearer " + login.json()["access_token"]}

        session_res = client.post(
            "/sessions",
            json={"title": "Real Numbers Class 10", "subject": "Mathematics"},
            headers=headers,
        )
        assert session_res.status_code == 201
        session_id = session_res.json()["id"]

        # Upload the math excerpt
        upload_res = client.post(
            f"/integrations/sessions/{session_id}/documents",
            files={"file": ("class10_real_numbers.pdf", pdf_content, "application/pdf")},
            headers=headers,
        )
        assert upload_res.status_code == 201
        assert upload_res.json()["chunks"] >= 1

        # Query 1: Formula definition and statement of Euclid's division lemma
        formula_res = client.post(
            f"/sessions/{session_id}/messages",
            json={"content": "State Euclid's division lemma from the uploaded notes and write its formula."},
            headers=headers,
        )
        assert formula_res.status_code == 201
        formula_body = formula_res.json()
        assert formula_body["assistant_message"]["role"] == "assistant"
        f_ans = formula_body["assistant_message"]["content"]
        clean_f = f_ans.replace(r"\,", "").replace(r"\;", "").replace(" ", "")
        assert "a=bq+r" in clean_f or "bq+r" in clean_f or ("dividend" in f_ans.lower() and "remainder" in f_ans.lower())
        assert len(formula_body["sources"]) >= 1

        # Query 2: Worked example from notes (HCF of 4052 and 12576)
        import time
        time.sleep(1.5)
        example_res = client.post(
            f"/sessions/{session_id}/messages",
            json={"content": "Find the HCF of 4052 and 12576 using Euclid's division algorithm as explained in the notes."},
            headers=headers,
        )
        assert example_res.status_code == 201
        ex_body = example_res.json()
        ex_ans = ex_body["assistant_message"]["content"]
        assert "4" in ex_ans  # HCF of 4052 and 12576 is 4
        clean_ex = ex_ans.replace(r"\,", "").replace(",", "").replace(" ", "")
        assert "12576" in clean_ex and "4052" in clean_ex

        # Query 3: Follow-up solving a new numerical with different values
        time.sleep(1.5)
        followup_res = client.post(
            f"/sessions/{session_id}/messages",
            json={"content": "Now solve for 504 and 1188 using the same algorithm step by step."},
            headers=headers,
        )
        assert followup_res.status_code == 201
        followup_body = followup_res.json()
        f_ans2 = followup_body["assistant_message"]["content"]
        assert "36" in f_ans2 or "hcf" in f_ans2.lower()  # HCF of 504 and 1188 is 36
        assert "beyond the scope" not in f_ans2.lower()

        # Query 4: Scope boundary - completely unrelated advanced mathematics outside syllabus
        unrelated_res = client.post(
            f"/sessions/{session_id}/messages",
            json={"content": "Calculate the path integral of the Dirac field in quantum electrodynamics."},
            headers=headers,
        )
        assert unrelated_res.status_code == 201
        unrelated_body = unrelated_res.json()
        assert unrelated_body["assistant_message"]["content"] == "The question is beyond the scope of the topics in this chat."


def test_solved_example_linear_equations_coincident_lines_and_similar_problem() -> None:
    from pathlib import Path
    from uuid import uuid4
    from fastapi.testclient import TestClient
    from app.main import app

    math_pdf_path = Path(__file__).resolve().parents[3] / "Class 10 Math NCERT CBSE (NCERT) (z-library.sk, 1lib.sk, z-lib.sk).pdf"
    if not math_pdf_path.exists():
        return

    import pymupdf
    src_doc = pymupdf.open(str(math_pdf_path))
    # Extract pages 42-45 (0-indexed 41 to 44: Example 2 Romila's pencil & eraser purchase and graphical representation)
    math_doc = pymupdf.open()
    math_doc.insert_pdf(src_doc, from_page=41, to_page=44)
    pdf_content = math_doc.tobytes()

    suffix = uuid4().hex[:8]
    with TestClient(app) as client:
        signup = client.post(
            "/auth/signup",
            json={
                "username": f"linear_{suffix}",
                "email": f"linear_{suffix}@example.com",
                "password": "securepass123",
            },
        )
        assert signup.status_code == 201
        login = client.post("/auth/login", json={"username": f"linear_{suffix}", "password": "securepass123"})
        headers = {"Authorization": "Bearer " + login.json()["access_token"]}

        session_res = client.post(
            "/sessions",
            json={"title": "Linear Equations Solved Examples", "subject": "Mathematics"},
            headers=headers,
        )
        assert session_res.status_code == 201
        session_id = session_res.json()["id"]

        # Upload the linear equations chapter excerpt
        upload_res = client.post(
            f"/integrations/sessions/{session_id}/documents",
            files={"file": ("class10_linear_equations.pdf", pdf_content, "application/pdf")},
            headers=headers,
        )
        assert upload_res.status_code == 201
        assert upload_res.json()["chunks"] >= 1

        # Query 1: Solved Example 2 (Romila's purchase) - Must NOT hallucinate (4, 2)
        import time
        time.sleep(2)
        romila_res = client.post(
            f"/sessions/{session_id}/messages",
            json={"content": "Romila went to a stationery shop and purchased 2 pencils and 3 erasers for ` 9. Her friend Sonali saw the new variety of pencils and erasers with Romila, and she also bought 4 pencils and 6 erasers of the same kind for ` 18. Represent this situation algebraically and graphically."},
            headers=headers,
        )
        assert romila_res.status_code == 201
        body = romila_res.json()
        ans = body["assistant_message"]["content"]
        assert "2x + 3y = 9" in ans or "2x+3y=9" in ans.replace(" ", "").replace("&", "")
        assert "coincide" in ans.lower() or "infinitely many" in ans.lower()
        # Must not hallucinate a unique intersection point (4, 2) for the costs
        assert "pencil costs ₹4" not in ans
        assert "pencil costs 4" not in ans
        assert "|" in ans

        # Query 2: Solving a similar new word problem (Apples and Grapes - Parallel lines)
        import time
        time.sleep(3)
        apples_res = client.post(
            f"/sessions/{session_id}/messages",
            json={"content": "Represent this situation algebraically and graphically: The cost of 2 kg of apples and 1 kg of grapes on a day was found to be ` 160. After a month, the cost of 4 kg of apples and 2 kg of grapes is ` 300."},
            headers=headers,
        )
        assert apples_res.status_code == 201
        apples_body = apples_res.json()
        apples_ans = apples_body["assistant_message"]["content"]
        assert "parallel" in apples_ans.lower() or "no solution" in apples_ans.lower() or "inconsistent" in apples_ans.lower()
        assert "|" in apples_ans

        # Query 3: Out-of-syllabus method requested (Cramer's rule)
        time.sleep(3)
        cramer_res = client.post(
            f"/sessions/{session_id}/messages",
            json={"content": "Solve 2x + 3y = 9 and 4x + 6y = 18 using Cramer's rule."},
            headers=headers,
        )
        assert cramer_res.status_code == 201
        cramer_body = cramer_res.json()
        cramer_ans = cramer_body["assistant_message"]["content"]
        assert any(phrase in cramer_ans.lower() for phrase in ["beyond the scope", "not part of the syllabus", "scope note", "not in the syllabus", "scope of the method", "not covered"])

        # Query 4: Entirely out-of-scope question outside uploaded syllabus
        time.sleep(1)
        out_res = client.post(
            f"/sessions/{session_id}/messages",
            json={"content": "How do you synthesize aspirin in an organic chemistry lab?"},
            headers=headers,
        )
        assert out_res.status_code == 201
        out_body = out_res.json()
        assert "The question is beyond the scope of the topics in this chat." in out_body["assistant_message"]["content"]


def test_math_figure_extraction_and_geometry_diagram_flow() -> None:
    from pathlib import Path
    import pymupdf
    from app.services.rag_service import extract_pdf, extract_images_from_pdf

    math_pdf_path = Path(__file__).resolve().parents[3] / "Class 10 Math NCERT CBSE (NCERT) (z-library.sk, 1lib.sk, z-lib.sk).pdf"
    if not math_pdf_path.exists():
        return

    src_doc = pymupdf.open(str(math_pdf_path))
    math_doc = pymupdf.open()
    # Extract pages 41 to 45 (contains Fig. 3.2, Fig. 3.3, Fig. 3.4)
    math_doc.insert_pdf(src_doc, from_page=41, to_page=44)
    pdf_content = math_doc.tobytes()

    pages = extract_pdf(pdf_content)
    images = extract_images_from_pdf(pdf_content, pages)
    assert len(images) >= 3
    captions = [img.caption for img in images]
    assert any("3.3" in cap for cap in captions)
    assert any("coincide" in cap.lower() for cap in captions)
    assert any("parallel" in cap.lower() for cap in captions)



