
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, JSON, func
from sqlmodel import Field, SQLModel


class Prediction(SQLModel, table=True):
    __tablename__ = "predictions"

    prediction_id: int | None = Field(default=None, primary_key=True)

    run_id: int = Field(
        foreign_key="model_runs.run_id",
        index=True,
    )

    input_data: dict[str, Any] = Field(sa_type=JSON)
    output_data: dict[str, Any] = Field(sa_type=JSON)

    status: str = Field(default="success", max_length=30)
    error_message: str | None = Field(default=None, max_length=2000)

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
        ),
    )
