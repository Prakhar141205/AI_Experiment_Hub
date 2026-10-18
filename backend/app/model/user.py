
from datetime import datetime

from sqlalchemy import Column, DateTime, String, func
from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    __tablename__ = "users"

    user_id: int | None = Field(default=None, primary_key=True)

    username: str = Field(
        sa_column=Column(String(255), unique=True, nullable=False)
    )

    email: str = Field(
        sa_column=Column(String(255), unique=True, nullable=False)
    )

    hashed_password: str = Field(
        sa_column=Column(String(255), nullable=False)
    )

    created_at: datetime | None = Field(
    default=None,
    sa_column=Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
)

    last_login: datetime | None = Field(
    default=None,
    sa_column=Column(DateTime(timezone=True), nullable=True),
)
    account_status: bool = Field(default=True, nullable=False)