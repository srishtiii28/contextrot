from datetime import UTC, datetime, timedelta

from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings

password_context = CryptContext(schemes=["pbkdf2_sha256", "bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return password_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_context.verify(plain_password, hashed_password)


def create_access_token(subject: str) -> str:
    settings = get_settings()
    expires_at = datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes)
    return jwt.encode({"sub": subject, "exp": expires_at}, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(credentials: HTTPAuthorizationCredentials | None) -> int:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    settings = get_settings()
    try:
        payload = jwt.decode(credentials.credentials, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        return int(payload["sub"])
    except (JWTError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token") from exc

