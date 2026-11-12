from fastapi import Depends, Header, HTTPException, status
from fastapi.security import APIKeyHeader
import hashlib
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import bearer_scheme, decode_access_token
from app.database.database import get_db
from app.database.models import User
from app.database.runtime_models import ApiKey
api_key_scheme = APIKeyHeader(name="X-API-Key",auto_error=False)


def current_user(db: Session = Depends(get_db), credentials=Depends(bearer_scheme), x_api_key: str | None = Depends(api_key_scheme)) -> User:
    if x_api_key:
        key = db.scalar(select(ApiKey).where(ApiKey.digest == hashlib.sha256(x_api_key.encode()).hexdigest(), ApiKey.revoked.is_(False)))
        if key is None:
            raise HTTPException(status_code=401, detail="Invalid or revoked API key")
        user_id = key.user_id
    else:
        user_id = decode_access_token(credentials)
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User no longer exists")
    return user

