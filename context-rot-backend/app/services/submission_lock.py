"""Serialize generation per session, including overlapping streaming requests."""
import time
from uuid import uuid4
from fastapi import HTTPException
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from app.database.database import SessionLocal
from app.database.runtime_models import SessionLease
from app.core.config import get_settings


def acquire(session_id):
    token=uuid4().hex;now=time.time()
    with SessionLocal() as db:
        if db.get(SessionLease,session_id) is None:
            db.add(SessionLease(session_id=session_id,expires_at=0))
            try:db.commit()
            except IntegrityError:db.rollback()  # Another request created the row first.
        result=db.execute(update(SessionLease).where(SessionLease.session_id==session_id,SessionLease.expires_at<now).values(token=token,expires_at=now+max(300,get_settings().model_timeout_seconds*2+90)))
        db.commit()
        if not result.rowcount:raise HTTPException(409,"Another question is being generated for this session")
    return token


def release(session_id,token):
    with SessionLocal() as db:
        db.execute(update(SessionLease).where(SessionLease.session_id==session_id,SessionLease.token==token).values(expires_at=0,token=None));db.commit()
