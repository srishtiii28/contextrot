from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import bearer_scheme, decode_access_token
from app.database.database import get_db
from app.database.models import User


def current_user(db: Session = Depends(get_db), credentials=Depends(bearer_scheme)) -> User:
    user_id = decode_access_token(credentials)
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User no longer exists")
    return user

