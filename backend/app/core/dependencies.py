
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel import Session, select

from app.core.config import settings
from app.core.tokens import ALGORITHM
from app.database.database import get_db
from app.model.user import User

bearer_scheme = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    session: Session = Depends(get_db),
) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.JWT_SECRET_KEY,
            algorithms=[ALGORITHM],
        )

        subject = payload.get("sub")

        if not isinstance(subject, str) or not subject.isdigit():
            raise unauthorized

        user_id = int(subject)

    except jwt.InvalidTokenError:
        raise unauthorized

    user = session.exec(
        select(User).where(User.user_id == user_id)
    ).first()

    if user is None or not user.account_status:
        raise unauthorized

    return user


