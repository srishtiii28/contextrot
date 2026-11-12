import json
import time
import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.database.models import User
from app.routes.dependencies import current_user
from app.schemas.message import ChatResponse, MessageCreate
from app.services.llm_service import generate_response
from app.services.pipeline import inject_fault, persist_answer, prepare
from app.services.session_service import owned_session
from app.services.submission_lock import acquire,release

router=APIRouter(prefix="/sessions",tags=["chat"])


@router.post("/{session_id}/messages",response_model=ChatResponse,status_code=201)
async def send_message(session_id:int,payload:MessageCreate,db:Session=Depends(get_db),user:User=Depends(current_user)):
    owned_session(db,session_id=session_id,user_id=user.id)
    start=time.perf_counter()
    lock=acquire(session_id)
    try:
        history,constraints,chunks=await prepare(db,session_id,payload)
        answer=await generate_response(user_message=payload.content,conversation_history=history,retrieved_context=chunks,constraints=constraints) if chunks else "This question is outside the available course material. Ask about a listed DBMS topic."
        answer=inject_fault(answer,payload.demo_fault) if chunks else answer
        return persist_answer(db,session_id,payload,answer,history,constraints,chunks,(time.perf_counter()-start)*1000)
    except (httpx.HTTPError,ValueError) as exc:
        db.rollback();raise HTTPException(503,"Generation or retrieval failed; check provider configuration") from exc
    finally:release(session_id,lock)


@router.post("/{session_id}/messages/stream",responses={200:{"content":{"text/event-stream":{}}}})
async def stream_message(session_id:int,payload:MessageCreate,db:Session=Depends(get_db),user:User=Depends(current_user)):
    owned_session(db,session_id=session_id,user_id=user.id)
    start=time.perf_counter()
    lock=acquire(session_id)
    try:history,constraints,chunks=await prepare(db,session_id,payload)
    except BaseException as exc:
        release(session_id,lock)
        if isinstance(exc,(httpx.HTTPError,ValueError)):raise HTTPException(503,"Retrieval failed") from exc
        raise
    async def stream():
        from app.services.streaming import generate_tokens
        def frame(kind,data):return "event: "+kind+"\ndata: "+json.dumps(data)+"\n\n"
        answer=""
        try:
            if not chunks:
                answer="This question is outside the available course material. Ask about a listed DBMS topic."
                yield frame("token",{"text":answer})
            elif payload.demo_fault:
                answer=inject_fault("",payload.demo_fault);yield frame("token",{"text":answer})
            else:
                async for text in generate_tokens(payload.content,history,chunks,constraints):
                    answer+=text;yield frame("token",{"text":text})
            result=persist_answer(db,session_id,payload,answer,history,constraints,chunks,(time.perf_counter()-start)*1000)
            yield frame("saved",ChatResponse(**result).model_dump(mode="json"))
        except Exception:
            db.rollback();yield frame("error",{"detail":"Generation failed; partial answer was not saved"})
        finally:release(session_id,lock)
    from starlette.background import BackgroundTask
    return StreamingResponse(stream(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"},background=BackgroundTask(release,session_id,lock))
