import asyncio
import hashlib
import secrets
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.core.security import decode_access_token
from app.database.database import SessionLocal, get_db
from app.database.models import Constraint, User
from app.database.runtime_models import ApiKey, CourseChunk, Event, Job
from app.routes.dependencies import current_user
from app.services.session_service import owned_session

router=APIRouter(tags=["platform API"])


class KeyCreate(BaseModel):
    name: str=Field(default="demo integration",min_length=1,max_length=100)


@router.post("/api-keys",status_code=201)
def create_key(payload:KeyCreate,db:Session=Depends(get_db),user:User=Depends(current_user)):
    secret="cr_"+secrets.token_urlsafe(32)
    key=ApiKey(user_id=user.id,name=payload.name,digest=hashlib.sha256(secret.encode()).hexdigest(),prefix=secret[:10])
    db.add(key);db.commit();db.refresh(key)
    return dict(id=key.id,name=key.name,key=secret,note="Shown once; only its hash is stored")


@router.get("/api-keys")
def keys(db:Session=Depends(get_db),user:User=Depends(current_user)):
    return [dict(id=k.id,name=k.name,prefix=k.prefix,revoked=k.revoked) for k in db.scalars(select(ApiKey).where(ApiKey.user_id==user.id))]


@router.delete("/api-keys/{key_id}")
def revoke(key_id:int,db:Session=Depends(get_db),user:User=Depends(current_user)):
    key=db.get(ApiKey,key_id)
    if not key or key.user_id!=user.id: raise HTTPException(404,"API key not found")
    key.revoked=True;db.commit();return {"revoked":True}


@router.get("/system/capabilities")
def capabilities():
    s=get_settings()
    return dict(provider=s.llm_provider,model=s.gemini_model if s.llm_provider=="gemini" else "offline-extractive",
        retrieval="teammate HTTP service" if s.rag_service_url else "local lexical demo baseline",
        detector="teammate HTTP service" if s.rot_service_url else "local rule-based demo baseline (unvalidated)",
        worker="embedded" if s.embedded_worker else "external (start separately)",
        demo_faults=s.enable_demo_faults and s.llm_provider in {"offline","mock"},max_correction_attempts=1,
        scope_policy="refuse when retrieval finds no matching course material")


@router.get("/course/topics")
def topics(db:Session=Depends(get_db),user:User=Depends(current_user)):
    return [dict(id=c.id,topic=c.topic,source=c.source,page=c.page) for c in db.scalars(select(CourseChunk))]


class Instruction(BaseModel):
    content:str=Field(min_length=1,max_length=2000)


@router.post("/sessions/{session_id}/constraints",status_code=201)
def constraint(session_id:int,payload:Instruction,db:Session=Depends(get_db),user:User=Depends(current_user)):
    owned_session(db,session_id=session_id,user_id=user.id)
    c=Constraint(session_id=session_id,constraint_type="instruction",content=payload.content)
    db.add(c);db.commit();db.refresh(c);return dict(id=c.id,content=c.content)


@router.get("/sessions/{session_id}/constraints")
def constraints(session_id:int,db:Session=Depends(get_db),user:User=Depends(current_user)):
    owned_session(db,session_id=session_id,user_id=user.id)
    return [dict(id=c.id,content=c.content) for c in db.scalars(select(Constraint).where(Constraint.session_id==session_id))]


def event_list(db,session_id,after):
    return [dict(id=e.id,kind=e.kind,payload=e.payload,created_at=e.created_at.isoformat()) for e in db.scalars(select(Event).where(Event.session_id==session_id,Event.id>after).order_by(Event.id).limit(100))]


@router.get("/sessions/{session_id}/events")
def events(session_id:int,after:int=0,db:Session=Depends(get_db),user:User=Depends(current_user)):
    owned_session(db,session_id=session_id,user_id=user.id)
    return event_list(db,session_id,after)


@router.get("/sessions/{session_id}/jobs")
def jobs(session_id:int,db:Session=Depends(get_db),user:User=Depends(current_user)):
    owned_session(db,session_id=session_id,user_id=user.id)
    return [dict(id=j.id,message_id=j.message_id,status=j.status,attempts=j.attempts,error=j.error) for j in db.scalars(select(Job).where(Job.session_id==session_id).order_by(Job.id))]


@router.post("/sessions/{session_id}/jobs/{job_id}/retry")
def retry_job(session_id:int,job_id:int,db:Session=Depends(get_db),user:User=Depends(current_user)):
    owned_session(db,session_id=session_id,user_id=user.id)
    j=db.get(Job,job_id)
    if not j or j.session_id!=session_id:raise HTTPException(404,"Job not found")
    if j.status!="failed" or j.attempts>=3:raise HTTPException(409,"Only failed jobs below three processing attempts can be retried")
    j.status="queued";j.error=None;j.claim=None;j.lease_until=0;db.commit()
    return dict(id=j.id,status=j.status)


@router.websocket("/sessions/{session_id}/events")
async def socket(ws:WebSocket,session_id:int,after:int=0):
    """First frame: {token: JWT} or {api_key: key}; secrets never go in URLs."""
    await ws.accept()
    try:
        credentials=await asyncio.wait_for(ws.receive_json(),timeout=10)
        with SessionLocal() as db:
            if credentials.get("api_key"):
                key=db.scalar(select(ApiKey).where(ApiKey.digest==hashlib.sha256(credentials["api_key"].encode()).hexdigest(),ApiKey.revoked.is_(False)))
                if not key:raise HTTPException(401,"Invalid key")
                user_id=key.user_id
            else:
                user_id=decode_access_token(HTTPAuthorizationCredentials(scheme="Bearer",credentials=credentials.get("token","")))
            owned_session(db,session_id=session_id,user_id=user_id)
        await ws.send_json({"kind":"connected","after":after})
        while True:
            with SessionLocal() as db:
                # Recheck access and expiry/revocation on established connections.
                if credentials.get("api_key"):
                    key=db.scalar(select(ApiKey).where(ApiKey.digest==hashlib.sha256(credentials["api_key"].encode()).hexdigest(),ApiKey.revoked.is_(False)))
                    if not key:raise HTTPException(401,"Revoked key")
                else:decode_access_token(HTTPAuthorizationCredentials(scheme="Bearer",credentials=credentials.get("token","")))
                batch=event_list(db,session_id,after)
            for e in batch:
                await ws.send_json(e);after=e["id"]
            try:
                message=await asyncio.wait_for(ws.receive_text(),timeout=0.4)
                if message=="ping":await ws.send_json({"kind":"pong"})
            except asyncio.TimeoutError:pass
    except (HTTPException,asyncio.TimeoutError,ValueError,TypeError,KeyError):
        await ws.close(code=1008)
    except WebSocketDisconnect:pass
