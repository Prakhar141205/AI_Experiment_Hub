
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.database.database import get_db
from app.schemas.user import UserCreate, UserPublic, UserLogin
from app.services.user_service import create_user, authenticate_user
from app.core.tokens import create_access_token
from app.core.dependencies import get_current_user
from app.model.user import User

router = APIRouter(prefix="/users", tags=["Users"])


@router.post(
    "/register",
    response_model=UserPublic,
    status_code=201,
)
def register_user(
    user_data: UserCreate,
    session: Session = Depends(get_db),
):
    try:
        return create_user(session, user_data)
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc



@router.post("/login")
def login_user(
    login_data: UserLogin,
    session: Session = Depends(get_db),
):
    user = authenticate_user(
        session,
        login_data.username_or_email,
        login_data.password,
    )

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid username/email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(subject=str(user.user_id))

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


@router.get("/me", response_model=UserPublic)
def read_current_user(
    current_user: User = Depends(get_current_user),
):
    return current_user
