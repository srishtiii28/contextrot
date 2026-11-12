import json
import time
import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import User
from app.routes.dependencies import current_user
from app.schemas.message import ChatResponse, MessageCreate
from app.services.llm_service import generate_response, normalize_subject_category
from app.services.pipeline import inject_fault, persist_answer, prepare
from app.services.session_service import owned_session
from app.services.submission_lock import acquire, release

router = APIRouter(prefix="/sessions", tags=["chat"])


@router.post("/{session_id}/messages", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
async def send_message(session_id: int, payload: MessageCreate, db: Session = Depends(get_db), user: User = Depends(current_user)):
    session = owned_session(db, session_id=session_id, user_id=user.id)
    start = time.perf_counter()
    lock = acquire(session_id)
    try:
        history, constraints, chunks = await prepare(db, session_id, payload)
        diagram_sources = []
        if chunks:
            user_text = payload.content or "Please solve the question shown in the attached image."
            kb_id = session.knowledge_base_id if session else None
            subject_category = normalize_subject_category(session.subject if session else "")

            diagram = None
            if subject_category != "mathematics":
                from app.services.rag_service import retrieve_diagram
                diagram = retrieve_diagram(
                    db,
                    session_id=session_id,
                    query=user_text,
                    knowledge_base_id=kb_id,
                    conversation_history=history,
                )
                if diagram:
                    diagram_sources.append({
                        "id": diagram.id,
                        "caption": diagram.caption,
                        "page": diagram.page,
                        "url": f"/images/{diagram.id}",
                    })

            answer = await generate_response(
                user_message=user_text,
                conversation_history=history,
                retrieved_context=chunks,
                constraints=constraints,
                subject=session.subject if session else None,
                image_data=payload.image_data,
            )
            answer = inject_fault(answer, payload.demo_fault)
            if diagram:
                diagram_md = f"![{diagram.caption}](/images/{diagram.id})\n*Figure: {diagram.caption} (Page {diagram.page})*"
                if f"/images/{diagram.id}" not in answer:
                    answer = f"{diagram_md}\n\n{answer}"
            from app.services.graph_service import process_message_graphs
            answer = process_message_graphs(answer)
        else:
            answer = "This question is outside the available course material. Ask about a listed syllabus topic."
        return persist_answer(db, session_id, payload, answer, history, constraints, chunks, (time.perf_counter() - start) * 1000, diagrams=diagram_sources)
    except (httpx.HTTPError, ValueError) as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="Generation or retrieval failed; check provider configuration") from exc
    finally:
        release(session_id, lock)


@router.post("/{session_id}/messages/stream", responses={200: {"content": {"text/event-stream": {}}}})
async def stream_message(session_id: int, payload: MessageCreate, db: Session = Depends(get_db), user: User = Depends(current_user)):
    owned_session(db, session_id=session_id, user_id=user.id)
    start = time.perf_counter()
    lock = acquire(session_id)
    try:
        history, constraints, chunks = await prepare(db, session_id, payload)
    except BaseException as exc:
        release(session_id, lock)
        if isinstance(exc, (httpx.HTTPError, ValueError)):
            raise HTTPException(status_code=503, detail="Retrieval failed") from exc
        raise

    async def stream():
        from app.services.streaming import generate_tokens

        def frame(kind, data):
            return "event: " + kind + "\ndata: " + json.dumps(data) + "\n\n"

        answer = ""
        try:
            if not chunks:
                answer = "This question is outside the available course material. Ask about a listed syllabus topic."
                yield frame("token", {"text": answer})
            elif payload.demo_fault:
                answer = inject_fault("", payload.demo_fault)
                yield frame("token", {"text": answer})
            else:
                async for text in generate_tokens(payload.content, history, chunks, constraints):
                    answer += text
                    yield frame("token", {"text": text})
            result = persist_answer(db, session_id, payload, answer, history, constraints, chunks, (time.perf_counter() - start) * 1000)
            yield frame("saved", ChatResponse(**result).model_dump(mode="json"))
        except Exception:
            db.rollback()
            yield frame("error", {"detail": "Generation failed; partial answer was not saved"})
        finally:
            release(session_id, lock)

    from starlette.background import BackgroundTask
    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        background=BackgroundTask(release, session_id, lock),
    )
