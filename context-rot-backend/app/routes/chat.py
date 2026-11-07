from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import AnalysisResult, Constraint, DocumentChunk, Message, User
from app.integrations.rag_client import RetrievalRequest, retrieve
from app.integrations.rot_client import analyze
from app.routes.dependencies import current_user
from app.schemas.analysis import AnalysisInput
from app.schemas.message import ChatResponse, MessageCreate
from app.services.graph_service import process_message_graphs
from app.services.llm_service import LLMConfigurationError, generate_response, normalize_subject_category
from app.services.rag_service import (
    _terms,
    check_scope_boundary,
    is_broad_study_request,
    is_follow_up_request,
    is_process_request,
    resolve_conversational_query,
    retrieve_diagram,
)
from app.services.session_service import owned_session

router = APIRouter(prefix="/sessions", tags=["chat"])


@router.post("/{session_id}/messages", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
async def send_message(session_id: int, payload: MessageCreate, db: Session = Depends(get_db), user: User = Depends(current_user)) -> ChatResponse:
    owned_session(db, session_id=session_id, user_id=user.id)
    user_text = (payload.content or "").strip() or ("Please solve the question shown in the attached image." if payload.image_data else "")
    user_message = Message(session_id=session_id, role="user", content=user_text, image_data=payload.image_data)
    db.add(user_message)
    db.flush()
    history = list(db.scalars(select(Message).where(Message.session_id == session_id).order_by(Message.created_at, Message.id)))
    session = owned_session(db, session_id=session_id, user_id=user.id)

    # 1. Maintain conversation continuity (resolve follow-up questions)
    history_dicts = [{"role": item.role, "content": item.content} for item in history[:-1]]
    resolved_message, retrieval_query = resolve_conversational_query(user_text, history_dicts)

    retrieved = await retrieve(
        RetrievalRequest(
            query=retrieval_query or "mathematics",
            session_id=session_id,
            knowledge_base_id=session.knowledge_base_id,
            top_k=16 if is_broad_study_request(retrieval_query) else (8 if is_process_request(retrieval_query) else 6),
        ),
        db,
    )

    # 2. Scope Guard: Syllabus boundary checking
    subject = (
        session.subject
        or (session.knowledge_base.subject if session.knowledge_base else None)
        or (session.knowledge_base.name if session.knowledge_base else None)
        or ""
    )
    subject_category = normalize_subject_category(
        subject,
        context_text=f"{user_text} {' '.join(c.text[:200] for c in retrieved.chunks[:3])}",
    )
    has_uploaded_docs = bool(
        session.documents or (session.knowledge_base and session.knowledge_base.documents)
    )
    chunk_objs = [
        DocumentChunk(
            document_id=c.document_id or 0,
            chunk_index=idx,
            text=c.text,
            embedding=b"",
            start_char=0,
            end_char=len(c.text),
        )
        for idx, c in enumerate(retrieved.chunks)
    ]
    if payload.image_data:
        # User attached image/screenshot: always accepted in-scope for solving
        in_scope = True
    else:
        in_scope = check_scope_boundary(user_text, chunk_objs, has_uploaded_docs=has_uploaded_docs, subject=subject)

    # 3. Diagram / Flowchart retrieval
    # For mathematics: Textbook plates are NEVER retrieved or displayed.
    # For biology / other subjects: Textbook plates are retrieved and explained as normal.
    diagram = None
    diagram_sources = []
    lowered_content = user_text.lower()

    if subject_category != "mathematics":
        diagram_descriptor_words = {
            "diagram", "diagrammatic", "representation", "figure", "fig", "image", "picture",
            "show", "draw", "drawing", "sketch", "tell", "me", "about", "above", "below",
            "previous", "earlier", "same", "that", "this", "it", "its", "them", "their",
            "give", "please", "display", "explain", "explanation", "label", "labels",
            "labelling", "labellings", "part", "parts", "what", "is", "are", "the", "of",
        }
        raw_content_terms = _terms(user_text)
        substantive_diagram_terms = raw_content_terms - diagram_descriptor_words

        if substantive_diagram_terms:
            diagram_query = user_text
        elif is_follow_up_request(user_text) or any(ref in lowered_content for ref in ("above", "that", "this", "it", "its", "same", "previous", "earlier")):
            diagram_query = retrieval_query
        else:
            diagram_query = user_text

        diagram = retrieve_diagram(
            db,
            session_id=session_id,
            query=diagram_query,
            knowledge_base_id=session.knowledge_base_id,
            conversation_history=history_dicts,
        )
        if diagram:
            diagram_sources.append({
                "id": diagram.id,
                "caption": diagram.caption,
                "page": diagram.page,
                "url": f"/images/{diagram.id}",
            })
            if subject_category == "biology":
                resolved_message += (
                    f"\n(Note: The textbook diagram '{diagram.caption}' from Page {diagram.page} is displayed to the student below your answer. "
                    f"Provide a comprehensive, exam-ready explanation of this diagram and its anatomical/physiological labellings based on the textbook notes. "
                    f"Systematically explain every labelled region, tissue layer, or structure (e.g. for transverse sections: outer cuticle/epidermis, cortex/hypodermis, endodermis, pericycle, vascular bundles, cambium, xylem, phloem, pith/medullary rays). "
                    f"If the diagram is a multi-part grouped plate containing (a) and (b), explain both parts thoroughly and highlight their key anatomical differences. "
                    f"NEVER say that the diagram is not available in the notes, because this diagram '{diagram.caption}' is actively displayed to the student right now.)"
                )
            else:
                resolved_message += (
                    f"\n(Note: The textbook figure '{diagram.caption}' from Page {diagram.page} is displayed to the student alongside your answer. "
                    f"Refer directly to this figure and explain its parts and scientific principles based on the notes. "
                    f"NEVER say that the diagram is not available in the notes.)"
                )
        elif has_uploaded_docs and any(term in lowered_content for term in ("diagram", "figure", "flowchart", "draw ", "sketch")):
            resolved_message += "\n(Note: No diagram for this specific topic exists in the uploaded notes. Do not generate ASCII art or text sketches. Explicitly state: 'The requested diagram is not available in the uploaded notes.')"
    else:
        # Mathematics: Diagrams are NEVER retrieved from textbook plates!
        # Dynamic Cartesian graphs and geometry rendering remain active:
        if has_uploaded_docs and any(term in lowered_content for term in ("diagram", "figure", "flowchart", "draw ", "sketch", "graph", "plot")):
            resolved_message += (
                "\n(Note: Do NOT draw ASCII art. Provide the verified coordinate table and output a ```graph or ```geometry JSON block "
                "so the visual graph/geometric figure is rendered directly for the student.)"
            )

    if payload.image_data:
        resolved_message += (
            "\n(Student Attached Image Instructions: The student has attached an image containing a question, geometric diagram, or mathematical problem. "
            "1. Carefully read and transcribe all text, equations, given measurements, and question statements visible in the image. "
            "2. If a geometry diagram or figure is present, identify all vertices, shapes, angles, markings, and relationships shown. "
            "3. State what is Given and what is To Find/To Prove. "
            "4. Solve the problem completely and accurately step by step using standard syllabus methods. "
            "5. If a graph or geometric construction is required or helpful, include the appropriate ```graph or ```geometry block.)"
        )

    if has_uploaded_docs and not in_scope:
        answer = "The question is beyond the scope of the topics in this chat."
    else:
        try:
            subject = (
                session.subject
                or (session.knowledge_base.subject if session.knowledge_base else None)
                or (session.knowledge_base.name if session.knowledge_base else None)
                or ""
            )
            answer = await generate_response(
                user_message=resolved_message,
                conversation_history=history_dicts,
                retrieved_context=retrieved.chunks,
                subject=subject,
                image_data=payload.image_data,
            )
            if diagram:
                is_explicit_diag = any(term in lowered_content for term in ("diagram", "figure", "fig", "draw", "sketch", "image", "picture", "illustration", "flowchart", "plate"))
                diagram_md = f"![{diagram.caption}](/images/{diagram.id})\n*Figure: {diagram.caption} (Page {diagram.page})*"
                if f"/images/{diagram.id}" not in answer:
                    if is_explicit_diag:
                        answer = f"{diagram_md}\n\n{answer}"
                    else:
                        answer = f"{answer}\n\n{diagram_md}"
            answer = process_message_graphs(answer)
            if answer.strip().strip("*#`_ ").rstrip(".").lower() == "the question is beyond the scope of the topics in this chat":
                answer = "The question is beyond the scope of the topics in this chat."
        except LLMConfigurationError as exc:
            db.rollback()
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    if answer == "The question is beyond the scope of the topics in this chat.":
        in_scope = False

    assistant_message = Message(session_id=session_id, role="assistant", content=answer, agent_type="llm")
    db.add(assistant_message)
    db.commit()
    db.refresh(user_message)
    db.refresh(assistant_message)

    constraints = list(db.scalars(select(Constraint).where(Constraint.session_id == session_id)))
    analysis_payload = AnalysisInput(
        session_id=session_id,
        message_id=assistant_message.id,
        current_question=user_text,
        current_answer=answer,
        conversation_history=history_dicts,
        constraints=[{"type": item.constraint_type, "content": item.content} for item in constraints],
    )
    rot_data = await analyze(analysis_payload)
    if has_uploaded_docs and not in_scope:
        rot_data["scope_score"] = 0.0
        rot_data["alert"] = "The question is beyond the scope of the topics in this chat."

    analysis = AnalysisResult(session_id=session_id, message_id=assistant_message.id, **rot_data)
    db.add(analysis)
    db.commit()

    unique_sources = []
    seen_sources = set()
    if in_scope:
        for chunk in retrieved.chunks:
            source_key = (chunk.source, chunk.page)
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            unique_sources.append(
                {"source": chunk.source, "page": chunk.page, "score": round(chunk.score, 4)}
            )
            if len(unique_sources) == 8:
                break

    return ChatResponse(
        user_message=user_message,
        assistant_message=assistant_message,
        sources=unique_sources,
        diagrams=diagram_sources,
    )
