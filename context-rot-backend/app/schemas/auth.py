from pydantic import BaseModel, EmailStr, Field

from app.schemas.common import Timestamped


class SignUpRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_]+$")
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(Timestamped):
    id: int
    username: str
    email: EmailStr

