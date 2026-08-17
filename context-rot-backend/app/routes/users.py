from fastapi import APIRouter, Depends

from app.database.models import User
from app.routes.dependencies import current_user
from app.schemas.auth import UserResponse

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserResponse)
def get_me(user: User = Depends(current_user)) -> User:
    return user

