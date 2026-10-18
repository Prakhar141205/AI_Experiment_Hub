
from datetime import datetime

from sqlalchemy import Column, DateTime, func
from sqlmodel import Field, SQLModel


class Dataset(SQLModel, table=True):
    __tablename__ = "datasets"

    dataset_id: int | None = Field(
        default=None,
        primary_key=True,
    )

    user_id: int = Field(
        foreign_key="users.user_id",
        index=True,
    )

    name: str = Field(max_length=100)

    original_filename: str = Field(max_length=255)

    file_path: str = Field(max_length=500)

    row_count: int = Field(ge=0)

    column_count: int = Field(ge=0)

    uploaded_at: datetime | None = Field(
    default=None,
    sa_column=Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
)
