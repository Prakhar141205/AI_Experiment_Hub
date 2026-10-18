
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, JSON, func
from sqlmodel import Field, SQLModel


class Experiment(SQLModel, table=True):
    __tablename__ = "experiments"

    experiment_id: int | None = Field(default=None, primary_key=True)

    user_id: int = Field(
        foreign_key="users.user_id",
        index=True,
    )
    dataset_id: int = Field(
        foreign_key="datasets.dataset_id",
        index=True,
    )

    name: str = Field(max_length=150)
    description: str | None = Field(default=None, max_length=1000)

    task_type: str = Field(max_length=30)
    target_column: str = Field(max_length=255)

    status: str = Field(default="pending", max_length=30)

    config: dict[str, Any] = Field(
        default_factory=dict,
        sa_type=JSON,
    )

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
        ),
    )
    updated_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
            onupdate=func.now(),
        ),
    )
