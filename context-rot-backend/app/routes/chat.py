import json
import re
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
        history, constraints, chunks, in_scope = await prepare(db, session_id, payload)
        diagram_sources = []
        user_text = payload.content or "Please solve the question shown in the attached image."
        warning_banner = (
            "> ⚠️ **Out of Syllabus:** This topic is outside your prescribed course material. "
            "The following answer is provided from general knowledge and is not verified against your syllabus notes.\n\n"
        )
        if in_scope and chunks:
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

            answer = await generate_response(
                user_message=user_text,
                conversation_history=history,
                retrieved_context=chunks,
                constraints=constraints,
                subject=session.subject if session else None,
                image_data=payload.image_data,
                in_scope=True,
            )
            answer = inject_fault(answer, payload.demo_fault)

            # Strip any inadvertent out-of-syllabus warning callout for verified in-scope questions
            answer = re.sub(r"^>\s*⚠️\s*\*\*Out of Syllabus:\*\*.*?\n\n", "", answer, flags=re.DOTALL | re.IGNORECASE).strip()

            if diagram:
                diagram_sources.append({
                    "id": diagram.id,
                    "caption": diagram.caption,
                    "page": diagram.page,
                    "url": f"/images/{diagram.id}",
                })
                diagram_md = f"![{diagram.caption}](/images/{diagram.id})\n*Figure: {diagram.caption} (Page {diagram.page})*"
                if f"/images/{diagram.id}" not in answer:
                    answer = f"{diagram_md}\n\n{answer}"
            from app.services.graph_service import process_message_graphs
            answer = process_message_graphs(answer)
        else:
            # Out of syllabus query: generate general knowledge answer with visual warning banner and suppressed diagrams
            raw_answer = await generate_response(
                user_message=user_text,
                conversation_history=history,
                retrieved_context=[],
                constraints=constraints,
                subject=session.subject if session else None,
                image_data=payload.image_data,
                in_scope=False,
            )
            if not raw_answer.strip().startswith("> ⚠️ **Out of Syllabus:**"):
                answer = f"{warning_banner}{raw_answer}"
            else:
                answer = raw_answer
            diagram_sources = []
            chunks = []

        return persist_answer(
            db,
            session_id,
            payload,
            answer,
            history,
            constraints,
            chunks if in_scope else [],
            (time.perf_counter() - start) * 1000,
            diagrams=diagram_sources,
            in_scope=in_scope,
        )
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
        history, constraints, chunks, in_scope = await prepare(db, session_id, payload)
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
            if not in_scope:
                warning_banner = (
                    "> ⚠️ **Out of Syllabus:** This topic is outside your prescribed course material. "
                    "The following answer is provided from general knowledge and is not verified against your syllabus notes.\n\n"
                )
                yield frame("token", {"text": warning_banner})
                answer += warning_banner
                async for text in generate_tokens(payload.content, history, [], constraints, in_scope=False):
                    answer += text
                    yield frame("token", {"text": text})
            elif payload.demo_fault:
                answer = inject_fault("", payload.demo_fault)
                yield frame("token", {"text": answer})
            else:
                async for text in generate_tokens(payload.content, history, chunks, constraints, in_scope=True):
                    answer += text
                    yield frame("token", {"text": text})
            result = persist_answer(
                db,
                session_id,
                payload,
                answer,
                history,
                constraints,
                chunks if in_scope else [],
                (time.perf_counter() - start) * 1000,
                in_scope=in_scope,
            )
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
