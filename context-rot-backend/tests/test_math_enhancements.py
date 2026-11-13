import pytest
from app.database.models import DocumentChunk
from app.services.rag_service import check_scope_boundary
from app.services.graph_service import (
    render_cartesian_graph,
    render_geometry_diagram,
    synthesize_graph_from_text,
    process_message_graphs,
    _get_line_points_from_equation,
)
from app.services.llm_service import _relevant_sentences


@pytest.fixture
def math_sample_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            document_id=49,
            chunk_index=85,
            text=(
                "Chapter 3: Pair of Linear Equations in Two Variables. "
                "An equation of the form ax + by + c = 0 where a, b, c are real numbers is a linear equation in two variables x and y. "
                "The graph of a pair of linear equations in two variables is represented by two lines. "
                "(i) If the lines intersect at a point, then that point gives the unique solution of the two equations. In this case, the pair of equations is consistent. "
                "(ii) If the lines are coincident, then there are infinitely many solutions — each point on the line being a solution. In this case, the pair of equations is dependent (consistent). "
                "(iii) If the lines are parallel, then the pair of equations has no solution. In this case, the pair of equations is inconsistent."
            ),
            embedding=b"",
            start_char=0,
            end_char=500,
        ),
        DocumentChunk(
            document_id=49,
            chunk_index=86,
            text=(
                "Example 2: Romila went to a stationery shop and purchased 2 pencils and 3 erasers for Rs 9. "
                "Her friend Sonali saw the new variety of pencils and erasers with Romila, and she also bought 4 pencils and 6 erasers of the same kind for Rs 18. "
                "Represent this situation algebraically and graphically. "
                "Algebraic representation: 2x + 3y = 9 and 4x + 6y = 18. "
                "Comparing ratios: a1/a2 = 2/4 = 1/2, b1/b2 = 3/6 = 1/2, c1/c2 = 9/18 = 1/2. "
                "Since a1/a2 = b1/b2 = c1/c2, the two lines coincide and the situation has infinitely many solutions."
            ),
            embedding=b"",
            start_char=0,
            end_char=500,
        ),
    ]


def test_scope_boundary_accepts_textbook_math(math_sample_chunks: list[DocumentChunk]) -> None:
    # Direct textbook questions must have ZERO false negatives
    assert check_scope_boundary(
        "Romila went to a stationery shop and purchased 2 pencils and 3 erasers for Rs 9. Represent this situation graphically.",
        math_sample_chunks,
        has_uploaded_docs=True,
        subject="Mathematics",
    ) is True
    assert check_scope_boundary(
        "State Euclid's division lemma and find HCF using division algorithm.",
        math_sample_chunks,
        has_uploaded_docs=True,
        subject="Mathematics",
    ) is True


def test_scope_boundary_accepts_unseen_numerical_questions(math_sample_chunks: list[DocumentChunk]) -> None:
    # Completely new/unseen numerical problems within Class 10 syllabus MUST be accepted
    assert check_scope_boundary(
        "Solve the pair of linear equations: 3x + 2y = 12 and 5x - 2y = 4",
        math_sample_chunks,
        has_uploaded_docs=True,
        subject="Mathematics",
    ) is True
    assert check_scope_boundary(
        "Solve: 71x - 97y = 133, 43x + 59y = 201",
        math_sample_chunks,
        has_uploaded_docs=True,
        subject="Mathematics",
    ) is True
    assert check_scope_boundary(
        "Find the coordinates of the vertices of the triangle formed by the lines x - y + 1 = 0 and 3x + 2y - 12 = 0 with the x-axis",
        math_sample_chunks,
        has_uploaded_docs=True,
        subject="Mathematics",
    ) is True
    assert check_scope_boundary(
        "Find the roots of the quadratic equation 2x^2 - 7x + 3 = 0",
        math_sample_chunks,
        has_uploaded_docs=True,
        subject="Mathematics",
    ) is True
    assert check_scope_boundary(
        "Find the HCF of 96 and 404 using prime factorisation",
        math_sample_chunks,
        has_uploaded_docs=True,
        subject="Mathematics",
    ) is True
    assert check_scope_boundary(
        "Find the 20th term of the AP: 2, 7, 12, ...",
        math_sample_chunks,
        has_uploaded_docs=True,
        subject="Mathematics",
    ) is True


def test_scope_boundary_accepts_cramer_rule_for_scope_note(math_sample_chunks: list[DocumentChunk]) -> None:
    # When a student asks to solve linear systems via Cramer's rule, it must NOT be rejected by the scope guard,
    # so that the solver can output the explicit **Scope Note:** and solve via in-syllabus substitution/elimination.
    assert check_scope_boundary(
        "Solve 2x + 3y = 9 and 4x + 6y = 18 using Cramer's rule",
        math_sample_chunks,
        has_uploaded_docs=True,
        subject="Mathematics",
    ) is True


def test_scope_boundary_rejects_out_of_syllabus_higher_math(math_sample_chunks: list[DocumentChunk]) -> None:
    # Higher-level university / calculus questions MUST be rejected
    assert check_scope_boundary(
        "Find the integral of x^3 * e^x dx",
        math_sample_chunks,
        has_uploaded_docs=True,
        subject="Mathematics",
    ) is False
    assert check_scope_boundary(
        "Compute the derivative of sin(x) with respect to x",
        math_sample_chunks,
        has_uploaded_docs=True,
        subject="Mathematics",
    ) is False
    assert check_scope_boundary(
        "Solve the differential equation dy/dx + 2y = 4x",
        math_sample_chunks,
        has_uploaded_docs=True,
        subject="Mathematics",
    ) is False
    assert check_scope_boundary(
        "Calculate the eigenvalues of the matrix [[1, 2], [3, 4]]",
        math_sample_chunks,
        has_uploaded_docs=True,
        subject="Mathematics",
    ) is False
    assert check_scope_boundary(
        "Evaluate the limit as x approaches 0 of sin(x)/x",
        math_sample_chunks,
        has_uploaded_docs=True,
        subject="Mathematics",
    ) is False


def test_scope_boundary_rejects_non_academic_queries(math_sample_chunks: list[DocumentChunk]) -> None:
    # Everyday non-academic queries must be strictly rejected
    assert check_scope_boundary(
        "how do I bake a chocolate cake at home?",
        math_sample_chunks,
        has_uploaded_docs=True,
        subject="Mathematics",
    ) is False
    assert check_scope_boundary(
        "what is the best stock to buy in the stock market?",
        math_sample_chunks,
        has_uploaded_docs=True,
        subject="Mathematics",
    ) is False
    assert check_scope_boundary(
        "how do I repair a broken car engine transmission?",
        math_sample_chunks,
        has_uploaded_docs=True,
        subject="Mathematics",
    ) is False


def test_cramer_rule_produces_scope_note(math_sample_chunks: list[DocumentChunk]) -> None:
    # Verify that requesting Cramer's rule emits the required Scope Note and in-syllabus substitution solution
    res = _relevant_sentences("Solve 2x + 3y = 9 and 4x + 6y = 18 using Cramer's rule", math_sample_chunks)
    assert "**Scope Note:**" in res
    assert "beyond the scope of the syllabus" in res
    assert "Substitution" in res or "infinitely many solutions" in res


def test_unseen_numerical_solution_generation(math_sample_chunks: list[DocumentChunk]) -> None:
    # Verify mock solving for unseen linear equation systems
    res = _relevant_sentences("Solve the pair of linear equations: 3x + 2y = 12 and 5x - 2y = 4", math_sample_chunks)
    assert "(2, 3)" in res
    assert "Elimination" in res or "Substitution" in res
    assert "| x | y |" in res or "```graph" in res


def test_cartesian_graph_rendering_and_coincident_overlay() -> None:
    spec = {
        "title": "Romila's Stationery Purchase",
        "xLabel": "Cost of Pencil (Rs)",
        "yLabel": "Cost of Eraser (Rs)",
        "nature": "coincident",
        "lines": [
            {"name": "2x + 3y = 9", "points": [{"x": 0, "y": 3}, {"x": 4.5, "y": 0}], "color": "#2563eb"},
            {"name": "4x + 6y = 18", "points": [{"x": 0, "y": 3}, {"x": 4.5, "y": 0}], "color": "#dc2626", "coincident": True},
        ],
        "points": [
            {"x": 0, "y": 3, "label": "A(0, 3)"},
            {"x": 4.5, "y": 0, "label": "B(4.5, 0)"},
        ],
    }
    svg = render_cartesian_graph(spec)
    assert "<svg" in svg and "</svg>" in svg
    assert "Coincident Overlay" in svg or "coincident" in svg.lower() or "#dc2626" in svg
    assert "(0, 3)" in svg
    assert "(4.5, 0)" in svg
    assert "<?xml" not in svg  # Clean embedded SVG


def test_geometry_diagram_rendering() -> None:
    spec = {
        "type": "geometry",
        "title": "Tangents from an External Point to a Circle",
        "elements": [
            {"type": "circle", "center": {"x": 5, "y": 5}, "radius": 3, "label": "Circle O"},
            {"type": "line", "points": [{"x": 10, "y": 5}, {"x": 5, "y": 8}], "label": "Tangent PT"},
            {"type": "point", "x": 10, "y": 5, "label": "P(10, 5)"},
        ],
    }
    svg = render_geometry_diagram(spec)
    assert "<svg" in svg and "</svg>" in svg
    assert "Circle O" in svg or "Tangent PT" in svg or "P(10, 5)" in svg


def test_synthesize_graph_from_equations() -> None:
    text = (
        "The system of equations 3x + 2y = 12 and 5x - 2y = 4 can be represented graphically. "
        "The lines intersect at the point (2, 3)."
    )
    synth = synthesize_graph_from_text(text)
    assert synth is not None
    assert len(synth["lines"]) >= 2
    assert any(p["x"] == 2 and p["y"] == 3 for p in synth["points"])

    # Ensure process_message_graphs embeds valid SVG from this text
    processed = process_message_graphs(text)
    assert "<svg" in processed and "</svg>" in processed


def test_equation_parser_supports_standard_and_zero_forms() -> None:
    pts1 = _get_line_points_from_equation("3x + 2y = 12")
    assert len(pts1) >= 2
    pts2 = _get_line_points_from_equation("x - y + 1 = 0")
    assert len(pts2) >= 2
    pts3 = _get_line_points_from_equation("3x + 2y - 12 = 0")
    assert len(pts3) >= 2


def test_ncert_graph_styling_with_arrows_origin_shaded_region() -> None:
    # Test NCERT Class 10 style graph with X, X', Y, Y' arrows, Origin O, and shaded triangle
    spec = {
        "title": "Triangle Bounded by x - y + 1 = 0, 3x + 2y - 12 = 0 and X-axis",
        "xLabel": "x",
        "yLabel": "y",
        "lines": [
            {"name": "x - y + 1 = 0", "points": [{"x": -1, "y": 0}, {"x": 2, "y": 3}], "color": "#2563eb"},
            {"name": "3x + 2y - 12 = 0", "points": [{"x": 4, "y": 0}, {"x": 2, "y": 3}], "color": "#dc2626"},
        ],
        "points": [
            {"x": -1, "y": 0, "label": "A(-1, 0)"},
            {"x": 4, "y": 0, "label": "B(4, 0)"},
            {"x": 2, "y": 3, "label": "C(2, 3)"},
        ],
        "shaded_region": [{"x": -1, "y": 0}, {"x": 4, "y": 0}, {"x": 2, "y": 3}],
        "shaded_label": "Shaded Triangular Region",
        "nature": "intersecting",
    }
    svg = render_cartesian_graph(spec)
    assert "<svg" in svg and "</svg>" in svg
    assert "Shaded Triangular Region" in svg or "Shaded" in svg
    assert "X" in svg
    assert "Y" in svg
    assert "(-1, 0)" in svg
    assert "(4, 0)" in svg
    assert "(2, 3)" in svg


def test_chapter_formulae_and_concepts_compilation(math_sample_chunks: list[DocumentChunk]) -> None:
    # Requesting all formulae and concepts from Chapter 3 must return the full ratio table, equations, and methods
    res = _relevant_sentences("Give all important formulae and concepts from chapter 3 pair of linear equations", math_sample_chunks)
    assert "General Form" in res or "a_1 x + b_1 y + c_1 = 0" in res
    assert "Ratio Comparison" in res
    assert "Intersecting lines" in res
    assert "Coincident lines" in res
    assert "Parallel lines" in res
    assert "Substitution" in res
    assert "Elimination" in res

    # Requesting all formulae from Chapter 1 Real Numbers
    res_ch1 = _relevant_sentences("Give all the important formulae and concepts from chapter 1 real numbers", math_sample_chunks)
    assert "Euclid's Division Lemma" in res_ch1
    assert "a = bq + r" in res_ch1
    assert "Fundamental Theorem of Arithmetic" in res_ch1
    assert "HCF" in res_ch1 and "LCM" in res_ch1


def test_practice_questions_generation_with_marking_scheme(math_sample_chunks: list[DocumentChunk]) -> None:
    # Requesting practice questions must return authentic 1, 2, 3, 4, and 6 mark questions with solutions and marking schemes
    res = _relevant_sentences("Generate practice questions of 1, 2, 3, 4, and 6 marks for mathematics with solutions and marking schemes", math_sample_chunks)
    assert "1-Mark Question" in res
    assert "2-Mark Question" in res
    assert "3-Mark Question" in res
    assert "4-Mark Question" in res
    assert "6-Mark Question" in res
    assert "Marking Scheme" in res
    assert "Case Study" in res


def test_textbook_example_diagram_retrieval() -> None:
    from app.database.database import SessionLocal
    from app.services.rag_service import retrieve_diagram

    db = SessionLocal()
    try:
        # 1. Romila's stationery example -> Fig. 3.3
        romila_diag = retrieve_diagram(
            db,
            session_id=-999,
            query="Romila went to a stationery shop and purchased 2 pencils and 3 erasers for Rs 9. Represent this situation graphically.",
            knowledge_base_id=4,
        )
        assert romila_diag is not None
        assert "3.3" in romila_diag.caption

        # 2. Akhila's fair example -> Fig. 3.2
        akhila_diag = retrieve_diagram(
            db,
            session_id=-999,
            query="Akhila goes to a fair with Rs 20 and wants to have rides on the Giant Wheel and play Hoopla. Represent this situation graphically.",
            knowledge_base_id=4,
        )
        assert akhila_diag is not None
        assert "3.2" in akhila_diag.caption

        # 3. Two railway tracks example -> Fig. 3.4
        rails_diag = retrieve_diagram(
            db,
            session_id=-999,
            query="Two rails are represented by the equations x + 2y - 4 = 0 and 2x + 4y - 12 = 0. Represent this situation geometrically.",
            knowledge_base_id=4,
        )
        assert rails_diag is not None
        assert "3.4" in rails_diag.caption

        # 4. Concentric circles example -> Fig. 10.8
        concentric_diag = retrieve_diagram(
            db,
            session_id=-999,
            query="Two concentric circles are of radii 5 cm and 3 cm. Find the length of the chord of the larger circle which touches the smaller circle.",
            knowledge_base_id=4,
        )
        assert concentric_diag is not None
        assert "10.8" in concentric_diag.caption

        # 5. Theorem 10.2 Tangents from external point -> Fig. 10.7
        tangent_diag = retrieve_diagram(
            db,
            session_id=-999,
            query="Prove that the lengths of tangents drawn from an external point to a circle are equal.",
            knowledge_base_id=4,
        )
        assert tangent_diag is not None
        assert "10.7" in tangent_diag.caption

        # 6. Basic Proportionality Theorem (Thales) -> Fig. 6.10
        thales_diag = retrieve_diagram(
            db,
            session_id=-999,
            query="State and prove Basic Proportionality Theorem (Thales theorem) for triangle ABC where DE || BC.",
            knowledge_base_id=4,
        )
        assert thales_diag is not None
        assert "6.10" in thales_diag.caption
    finally:
        db.close()


def test_graphical_solving_does_not_attach_unrelated_diagram() -> None:
    from app.database.database import SessionLocal
    from app.services.rag_service import retrieve_diagram

    db = SessionLocal()
    try:
        # Standard graphical equation check/solve queries must NOT retrieve unrelated textbook raster plates
        q1 = "Check graphically whether the pair of equations x + 3y = 6 (1) and 2x - 3y = 12 (2) is consistent. If so, solve them graphically."
        diag1 = retrieve_diagram(db, session_id=-999, query=q1, knowledge_base_id=4)
        assert diag1 is None, f"Expected None for graphical math problem, but got: {diag1.id} ({diag1.caption})"

        q2 = "Solve the following pair of linear equations graphically: 2x + y = 6, 2x - y = 2"
        diag2 = retrieve_diagram(db, session_id=-999, query=q2, knowledge_base_id=4)
        assert diag2 is None, f"Expected None for graphical math problem, but got: {diag2.id} ({diag2.caption})"
    finally:
        db.close()


def test_cyclic_quadrilateral_figure_and_no_mismatched_fallback() -> None:
    from app.database.database import SessionLocal
    from app.services.rag_service import retrieve_diagram

    db = SessionLocal()
    try:
        # 1. Cyclic quadrilateral query MUST retrieve Fig. 3.7 from Chapter 3, NOT Fig. 10.12 from Chapter 10
        q = "ABCD is a cyclic quadrilateral (see Fig. 3.7).\nFind the angles of the cyclic quadrilateral."
        diag = retrieve_diagram(db, session_id=-999, query=q, knowledge_base_id=4)
        assert diag is not None, "Expected Fig. 3.7 to be retrieved, but got None"
        assert "3.7" in diag.caption, f"Expected Fig. 3.7, but got {diag.caption}"
        assert "10.12" not in diag.caption, f"Mismatched Fig. 10.12 returned! {diag.caption}"

        # 2. Query asking for a non-existent figure number must return None rather than falling back to an unrelated figure
        q_missing = "Refer to Fig. 99.8 and find the quadrilateral properties."
        diag_missing = retrieve_diagram(db, session_id=-999, query=q_missing, knowledge_base_id=4)
        assert diag_missing is None, f"Expected None when figure number is not in notes, but got: {diag_missing.caption}"
    finally:
        db.close()


def test_message_create_accepts_image_data_and_validates() -> None:
    from app.schemas.message import MessageCreate

    # Text only
    m1 = MessageCreate(content="Solve x + y = 2")
    assert m1.content == "Solve x + y = 2"
    assert m1.image_data is None

    # Image only (content empty)
    m2 = MessageCreate(content="", image_data="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...")
    assert m2.image_data.startswith("data:image/png")
    assert m2.content == ""

    # Both text and image
    m3 = MessageCreate(content="Find angle A", image_data="data:image/jpeg;base64,...")
    assert m3.content == "Find angle A"
    assert m3.image_data is not None

    # Neither text nor image should raise ValidationError
    with pytest.raises(Exception):
        MessageCreate(content="", image_data=None)


@pytest.mark.anyio
async def test_generate_response_handles_image_data(monkeypatch) -> None:
    from app.services.llm_service import generate_response
    from app.integrations.rag_client import RetrievedChunk

    captured_payload = {}

    class DummyResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {
                "choices": [{
                    "message": {"content": "The question is solved: x = 5, y = 3."},
                    "finish_reason": "stop"
                }]
            }

    class DummyClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, url, headers=None, json=None):
            captured_payload.update(json or {})
            return DummyResponse()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: DummyClient())
    from app.core.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "gsk-mock-key")

    dummy_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAALUlEQVRYR+3QQREAAAzCMPtnukf6wwJy1E3SnR4BAgQIECBAgAABAgQIECBA4L3AB1kBAe1yWksAAAAASUVORK5CYII="
    ans = await generate_response(
        user_message="Solve the attached question",
        conversation_history=[],
        retrieved_context=[RetrievedChunk(text="Linear equations: ax + by = c", page=1, score=0.9, source="notes.pdf")],
        subject="Mathematics",
        image_data=dummy_image,
    )
    assert "x = 5" in ans
    assert captured_payload["model"] == "qwen/qwen3.8-27b"
    user_msgs = [m for m in captured_payload["messages"] if m["role"] == "user"]
    assert len(user_msgs) == 1
    content_list = user_msgs[0]["content"]
    assert isinstance(content_list, list)
    assert any(item.get("type") == "image_url" and item.get("image_url", {}).get("url") == dummy_image for item in content_list)



