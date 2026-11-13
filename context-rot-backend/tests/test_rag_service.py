from app.services.rag_service import (
    ExtractedPage,
    _keyword_overlap,
    _phrase_overlap,
    check_scope_boundary,
    chunk_pages,
    cosine_similarity,
    embed,
    is_broad_study_request,
    is_follow_up_request,
    is_process_request,
    resolve_conversational_query,
    retrieve_diagram,
)
from app.services.llm_service import _clean_model_formatting, _has_duplicate_mcq_options, _is_four_mark_request, _response_guidance


def test_chunking_preserves_page_metadata_and_overlap() -> None:
    chunks = chunk_pages([ExtractedPage(page=3, text="one two three four five six")], chunk_size=15, overlap=5)

    assert chunks[0][1] == 3
    assert len(chunks) >= 2
    assert chunks[1][2] < chunks[0][3]


def test_embeddings_are_normalized_and_similar_text_scores_higher() -> None:
    query = embed("photosynthesis converts light energy")
    related = embed("light energy conversion in photosynthesis")
    unrelated = embed("database authentication sessions")

    assert cosine_similarity(query, query) == 1.0
    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


def test_keyword_overlap_prioritizes_an_exact_topic_match() -> None:
    assert _keyword_overlap("is carbohydrates mentioned in the notes?", "Carbohydrates are broken down into glucose.") == 1
    assert _keyword_overlap("is carbohydrates mentioned in the notes?", "The chapter introduces cellular respiration.") == 0
    assert _keyword_overlap("is carbohydrates mentioned in the notes?", "Carbohydrate metabolism is discussed.") == 1
    assert _phrase_overlap("is active transport present in the notes?", "The process is called active transport.") == 1


def test_chunking_does_not_split_words_when_a_boundary_is_available() -> None:
    chunks = chunk_pages(
        [ExtractedPage(page=1, text="alpha beta gamma delta epsilon zeta")],
        chunk_size=18,
        overlap=4,
    )
    assert all(chunk[0].split()[0] in {"alpha", "beta", "gamma", "delta", "epsilon", "zeta"} for chunk in chunks)


def test_broad_study_requests_use_revision_retrieval() -> None:
    assert is_broad_study_request("tell me important question answers from the biomolecules chapter")
    assert not is_broad_study_request("what is the function of carbohydrates?")
    assert is_follow_up_request("give me a few more 6 mark answers")
    assert not is_follow_up_request("what is the function of carbohydrates?")
    assert is_process_request("what are the steps of glycolysis?")
    assert not is_process_request("is glycolysis mentioned?")


def test_answer_style_detection_and_format_cleanup() -> None:
    assert _is_four_mark_request("Give a 4 mark answer on mitosis")
    assert _is_four_mark_request("Write a four-mark answer")
    assert not _is_four_mark_request("What are the four phases?")
    assert _clean_model_formatting("**Question:**\n*Answer:*") == "**Question:**\n*Answer:*"
    assert "one word" in _response_guidance("Give one-word answers").lower()
    assert "mcq" in _response_guidance("Create MCQs from this chapter").lower()
    assert "220-320" in _response_guidance("Give 6 mark answers").lower()
    assert "comprehensive" in _response_guidance("Give all important questions").lower()
    assert "connected prose" in _response_guidance("Give a 4 mark answer on enzymes").lower()
    assert "two or three connected paragraphs" in _response_guidance("Give a descriptive answer on enzymes").lower()
    assert _has_duplicate_mcq_options("A. Ribosome\nB. Ribosome\nC. Nucleus\nD. Cytoplasm")
    assert not _has_duplicate_mcq_options("A. Ribosome\nB. Nucleus\nC. Cytoplasm\nD. Ribozyme")


def test_semantic_retrieval_and_scope_merging() -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database.database import Base
    from app.database.models import Document, DocumentChunk
    from app.services.rag_service import retrieve_chunks, serialize_embedding, embed

    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(test_engine)
    TestingSession = sessionmaker(bind=test_engine)
    db = TestingSession()

    # Create session document
    doc1 = Document(session_id=1, filename="biology_cell.pdf", content_type="application/pdf")
    doc1.chunks = [
        DocumentChunk(
            chunk_index=0,
            text="Mitochondria are double-membrane-bound organelles that generate most of the chemical energy needed to power the biochemical reactions, stored as adenosine triphosphate (ATP).",
            page=12,
            start_char=0,
            end_char=180,
            embedding=serialize_embedding(embed("Mitochondria generate ATP energy.")),
        ),
        DocumentChunk(
            chunk_index=1,
            text="Endoplasmic reticulum is divided into rough and smooth. Rough endoplasmic reticulum is involved in protein synthesis and folding.",
            page=13,
            start_char=0,
            end_char=150,
            embedding=serialize_embedding(embed("Rough endoplasmic reticulum synthesizes proteins.")),
        ),
    ]
    db.add(doc1)

    # Create knowledge base document linked to the session's library
    doc2 = Document(knowledge_base_id=5, filename="syllabus_core.pdf", content_type="application/pdf")
    doc2.chunks = [
        DocumentChunk(
            chunk_index=0,
            text="Cell division occurs through mitosis for somatic cells and meiosis for gamete production.",
            page=45,
            start_char=0,
            end_char=110,
            embedding=serialize_embedding(embed("Cell division mitosis meiosis.")),
        )
    ]
    db.add(doc2)
    db.commit()

    # Query with semantic concept for energy generation
    retrieved = retrieve_chunks(
        db,
        session_id=1,
        query="What organelle produces ATP energy for the cell?",
        top_k=3,
        knowledge_base_id=5,
    )
    assert len(retrieved) >= 1
    assert "Mitochondria" in retrieved[0].text

    # Query for cell division from the knowledge base doc
    retrieved_kb = retrieve_chunks(
        db,
        session_id=1,
        query="How do somatic cells divide and reproduce?",
        top_k=3,
        knowledge_base_id=5,
    )
    assert any("Cell division" in c.text for c in retrieved_kb)
    db.close()


def test_resolve_conversational_query_preserves_topic() -> None:
    history = [
        {"role": "user", "content": "give me 5 questions from biomolecules chapter"},
        {"role": "assistant", "content": "Here are 5 questions on biomolecules..."},
    ]
    resolved, retrieval_query = resolve_conversational_query("give me more questions", history)
    assert "biomolecules" in resolved.lower()
    assert "biomolecules" in retrieval_query.lower()

    # Short follow-up without questions keyword
    resolved_short, retrieval_short = resolve_conversational_query("more please", history)
    assert "biomolecules" in resolved_short.lower()
    assert "biomolecules" in retrieval_short.lower()

    # Brand new independent question does not alter query
    new_query = "explain the laws of thermodynamics in physics"
    resolved_new, retrieval_new = resolve_conversational_query(new_query, history)
    assert resolved_new == new_query
    assert retrieval_new == new_query


def test_check_scope_boundary_accepts_in_scope_and_rejects_out_of_scope() -> None:
    from app.database.models import DocumentChunk

    sample_chunks = [
        DocumentChunk(
            document_id=1,
            chunk_index=0,
            text="Photosynthesis in green plants converts carbon dioxide and water into carbohydrates and oxygen using sunlight absorbed by chlorophyll.",
            embedding=b"",
            start_char=0,
            end_char=140,
        ),
        DocumentChunk(
            document_id=1,
            chunk_index=1,
            text="Cellular respiration in mitochondria breaks down glucose to release energy in the form of ATP through glycolysis and the citric acid cycle.",
            embedding=b"",
            start_char=0,
            end_char=140,
        ),
    ]

    # In-scope biology questions
    assert check_scope_boundary("what is the role of chlorophyll in photosynthesis?", sample_chunks, has_uploaded_docs=True) is True
    assert check_scope_boundary("how is ATP produced in cellular respiration?", sample_chunks, has_uploaded_docs=True) is True

    # Conversational turns & follow-ups are permitted
    assert check_scope_boundary("hi", sample_chunks, has_uploaded_docs=True) is True
    assert check_scope_boundary("thank you", sample_chunks, has_uploaded_docs=True) is True
    assert check_scope_boundary("give me more questions", sample_chunks, has_uploaded_docs=True) is True

    # When no documents are uploaded, general questions are allowed
    assert check_scope_boundary("tell me about French literature", [], has_uploaded_docs=False) is True

    # When documents are uploaded, completely unrelated questions outside syllabus must be rejected
    assert check_scope_boundary("how do I bake a chocolate cake at home?", sample_chunks, has_uploaded_docs=True) is False
    assert check_scope_boundary("what is the best stock to buy in the stock market?", sample_chunks, has_uploaded_docs=True) is False
    assert check_scope_boundary("how do I repair a broken car engine transmission?", sample_chunks, has_uploaded_docs=True) is False


def test_retrieve_diagram_matches_caption() -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database.database import Base
    from app.database.models import Document, DocumentImage

    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(test_engine)
    TestingSession = sessionmaker(bind=test_engine)
    db = TestingSession()

    doc = Document(session_id=20, filename="biology_class11.pdf", content_type="application/pdf")
    db.add(doc)
    db.flush()

    img1 = DocumentImage(
        document_id=doc.id,
        page=15,
        image_filename="page_15_img_0.png",
        caption="Figure 8.2: Diagrammatic representation of a Plant Cell with chloroplast and cell wall",
        image_data=b"fake-png-data-1",
    )
    img2 = DocumentImage(
        document_id=doc.id,
        page=34,
        image_filename="page_34_img_0.png",
        caption="Figure 10.1: Stages of Mitosis and Cell Division Cycle",
        image_data=b"fake-png-data-2",
    )
    db.add_all([img1, img2])
    db.commit()

    # 1. Requesting plant cell diagram
    found1 = retrieve_diagram(db, session_id=20, query="Can you show me the diagram of a plant cell?")
    assert found1 is not None
    assert found1.id == img1.id
    assert "Plant Cell" in found1.caption

    # 2. Requesting mitosis cycle flowchart
    found2 = retrieve_diagram(db, session_id=20, query="Show the flowchart or stages of mitosis")
    assert found2 is not None
    assert found2.id == img2.id
    assert "Mitosis" in found2.caption

    # 3. Non-diagram query returns None
    no_diagram = retrieve_diagram(db, session_id=20, query="Explain the chemical composition of cell wall")
    assert no_diagram is None

    # 4. Unrelated diagram query returns None (does not hallucinate another diagram)
    unrelated_diag = retrieve_diagram(db, session_id=20, query="Can you show me the diagram of an airplane engine?")
    assert unrelated_diag is None

    db.close()


def test_exact_diagram_extraction_and_heart_retrieval() -> None:
    from pathlib import Path
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database.database import Base
    from app.database.models import Document, DocumentImage
    from app.services.rag_service import extract_images_from_pdf, extract_pdf

    ncert_path = Path(__file__).resolve().parents[3] / "11 Biology NCERT (NCERT) (z-library.sk, 1lib.sk, z-lib.sk).pdf"
    if not ncert_path.exists():
        return

    import pymupdf

    # Extract just page 210 (0-indexed 209, heart section) as a self-contained PDF in memory
    src_doc = pymupdf.open(str(ncert_path))
    heart_doc = pymupdf.open()
    heart_doc.insert_pdf(src_doc, from_page=209, to_page=209)
    pdf_bytes = heart_doc.tobytes()

    pages = extract_pdf(pdf_bytes)
    extracted_images = extract_images_from_pdf(pdf_bytes, pages)

    assert len(extracted_images) >= 1
    heart_img = next((img for img in extracted_images if "heart" in img.caption.lower()), None)
    assert heart_img is not None
    assert "Figure 15.2" in heart_img.caption
    assert "human heart" in heart_img.caption.lower()
    assert heart_img.data.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(heart_img.data) > 50000  # High-resolution rendered PNG

    # Test DB retrieval
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(test_engine)
    TestingSession = sessionmaker(bind=test_engine)
    db = TestingSession()

    doc = Document(session_id=101, filename="ncert_heart.pdf", content_type="application/pdf")
    doc.images = [
        DocumentImage(
            page=heart_img.page,
            image_filename=heart_img.filename,
            caption=heart_img.caption,
            image_data=heart_img.data,
        )
    ]
    db.add(doc)
    db.commit()

    # Query matching subtitle
    found_heart = retrieve_diagram(db, session_id=101, query="diagram of a heart")
    assert found_heart is not None
    assert "Section of a human heart" in found_heart.caption

    # Direct figure reference
    found_fig = retrieve_diagram(db, session_id=101, query="Show Figure 15.2 from the notes")
    assert found_fig is not None
    assert found_fig.id == found_heart.id

    # Completely unrelated diagram request returns None
    unrelated = retrieve_diagram(db, session_id=101, query="show me the diagram of a blast furnace")
    assert unrelated is None

    db.close()


def test_multipart_diagram_extraction_stem_and_root() -> None:
    from pathlib import Path
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database.database import Base
    from app.database.models import Document, DocumentImage
    from app.services.rag_service import extract_images_from_pdf, extract_pdf
    from PIL import Image
    import io

    ncert_path = Path(__file__).resolve().parents[3] / "11 Biology NCERT (NCERT) (z-library.sk, 1lib.sk, z-lib.sk).pdf"
    if not ncert_path.exists():
        return

    import pymupdf

    # Extract Page 87 (0-indexed 86, Figure 6.4: Dicot & Monocot stem)
    src_doc = pymupdf.open(str(ncert_path))
    stem_doc = pymupdf.open()
    stem_doc.insert_pdf(src_doc, from_page=86, to_page=86)
    pdf_bytes = stem_doc.tobytes()

    pages = extract_pdf(pdf_bytes)
    extracted = extract_images_from_pdf(pdf_bytes, pages)

    assert len(extracted) >= 1
    stem_img = next((img for img in extracted if "stem" in img.caption.lower()), None)
    assert stem_img is not None
    assert "Figure 6.4" in stem_img.caption
    assert "(a) Dicot" in stem_img.caption
    assert "(b) Monocot" in stem_img.caption

    # Check that the image captured the full combined height (both (a) and (b)), not just (b)
    im = Image.open(io.BytesIO(stem_img.data))
    assert im.height > 900  # Combined height with both sub-diagrams is > 900 px (at 150 DPI)
    assert im.width > 800

    # Test retrieval for both "dicot stem" and "monocot stem"
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(test_engine)
    TestingSession = sessionmaker(bind=test_engine)
    db = TestingSession()

    doc = Document(session_id=102, filename="ncert_stem.pdf", content_type="application/pdf")
    doc.images = [
        DocumentImage(
            page=stem_img.page,
            image_filename=stem_img.filename,
            caption=stem_img.caption,
            image_data=stem_img.data,
        )
    ]
    db.add(doc)
    db.commit()

    # Query for dicot stem returns Figure 6.4
    found_dicot = retrieve_diagram(db, session_id=102, query="diagram of dicot stem")
    assert found_dicot is not None
    assert found_dicot.id == doc.images[0].id

    # Query for monocot stem returns the same combined Figure 6.4
    found_monocot = retrieve_diagram(db, session_id=102, query="diagram of monocot stem")
    assert found_monocot is not None
    assert found_monocot.id == doc.images[0].id

    db.close()


def test_math_formula_and_symbol_extraction() -> None:
    from pathlib import Path
    from app.services.rag_service import extract_pdf, chunk_pages

    math_pdf_path = Path(__file__).resolve().parents[3] / "Class 10 Math NCERT CBSE (NCERT) (z-library.sk, 1lib.sk, z-lib.sk).pdf"
    if not math_pdf_path.exists():
        return

    import pymupdf
    src_doc = pymupdf.open(str(math_pdf_path))
    # Extract Page 5 (0-indexed 4, Euclid's division algorithm worked example 1)
    sample_doc = pymupdf.open()
    sample_doc.insert_pdf(src_doc, from_page=4, to_page=4)
    pdf_bytes = sample_doc.tobytes()

    pages = extract_pdf(pdf_bytes)
    assert len(pages) == 1
    text = pages[0].text
    # Verify math symbols are accurately extracted without mojibake or font errors
    assert "Example 1" in text
    assert "4052" in text and "12576" in text
    assert "×" in text or "*" in text
    assert "≠" in text or "!=" in text
    assert "HCF" in text

    # Verify chunking preserves step structure
    chunks = chunk_pages(pages, chunk_size=1200, overlap=150)
    assert len(chunks) >= 1
    # Check that Step 1 equation is intact
    assert any("12576 = 4052" in c[0] for c in chunks)


def test_numerical_intent_detection_and_response_guidance() -> None:
    from app.services.rag_service import is_process_request, is_follow_up_request
    from app.services.llm_service import _response_guidance

    # Numerical problem solving intent
    assert is_process_request("Use Euclid's algorithm to find the HCF of 4052 and 12576")
    assert is_process_request("Solve this numerical step by step")
    assert is_process_request("Calculate the value using the formula")
    assert is_process_request("Find the roots of the quadratic equation")
    assert is_process_request("Derive the formula for kinetic energy")

    # Follow up numerical query
    assert is_follow_up_request("Now solve for 504 and 1188")
    assert is_follow_up_request("Give a similar problem")
    assert is_follow_up_request("Solve another example using the same technique")
    assert is_follow_up_request("What about for x = 10?")

    # Response guidance for math/numericals enforces step-by-step and LaTeX
    guidance = _response_guidance("Solve the numerical problem for 504 and 1188")
    assert "step-by-step" in guidance.lower()
    assert "latex" in guidance.lower()
    assert "formula" in guidance.lower()



