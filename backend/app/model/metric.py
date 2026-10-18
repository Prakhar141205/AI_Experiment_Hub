
from datetime import datetime

from sqlalchemy import Column, DateTime, func
from sqlmodel import Field, SQLModel


class ExperimentMetric(SQLModel, table=True):
    __tablename__ = "experiment_metrics"

    metric_id: int | None = Field(default=None, primary_key=True)

    run_id: int = Field(
        foreign_key="model_runs.run_id",
        index=True,
    )

    metric_name: str = Field(max_length=100)
    metric_value: float

    dataset_split: str = Field(
        default="test",
        max_length=30,
    )

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
        ),
    )
