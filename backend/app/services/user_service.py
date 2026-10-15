
from sqlalchemy import or_
from sqlmodel import Session, select

from app.core.security import hash_password, verify_password
from app.model.user import User
from app.schemas.user import UserCreate


def create_user(session: Session, user_data: UserCreate) -> User:
    # Check whether the username or email is already registered.
    statement = select(User).where(
        or_(
            User.username == user_data.username,
            User.email == str(user_data.email),
        )
    )

    existing_user = session.exec(statement).first()

    if existing_user:
        raise ValueError("Username or email is already registered")

    # Never store the plaintext password.
    user = User(
        username=user_data.username,
        email=str(user_data.email),
        hashed_password=hash_password(user_data.password),
    )

    session.add(user)
    session.commit()
    session.refresh(user)

    return user


def authenticate_user(
    session: Session,
    username_or_email: str,
    password: str,
) -> User | None:
    statement = select(User).where(
        or_(
            User.username == username_or_email,
            User.email == username_or_email,
        )
    )

    user = session.exec(statement).first()

    if user is None:
        return None

    if not user.account_status:
        return None

    if not verify_password(password, user.hashed_password):
        return None

    return user
