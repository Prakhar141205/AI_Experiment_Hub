
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, JSON, func
from sqlmodel import Field, SQLModel


class ModelRun(SQLModel, table=True):
    __tablename__ = "model_runs"

    run_id: int | None = Field(default=None, primary_key=True)

    experiment_id: int = Field(
        foreign_key="experiments.experiment_id",
        index=True,
    )

    algorithm: str = Field(max_length=100)
    model_version: str | None = Field(default=None, max_length=100)

    hyperparameters: dict[str, Any] = Field(
        default_factory=dict,
        sa_type=JSON,
    )

    status: str = Field(default="pending", max_length=30)

    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    duration_seconds: float | None = Field(default=None, ge=0)

    error_message: str | None = Field(default=None, max_length=2000)

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
        ),
    )
