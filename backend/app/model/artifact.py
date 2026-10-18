
from datetime import datetime

from sqlalchemy import Column, DateTime, func
from sqlmodel import Field, SQLModel


class Artifact(SQLModel, table=True):
    __tablename__ = "artifacts"

    artifact_id: int | None = Field(default=None, primary_key=True)

    experiment_id: int = Field(
        foreign_key="experiments.experiment_id",
        index=True,
    )

    # Optional: an artifact may belong to a specific training run.
    run_id: int | None = Field(
        default=None,
        foreign_key="model_runs.run_id",
        index=True,
    )

    name: str = Field(max_length=255)
    artifact_type: str = Field(max_length=50)
    file_path: str = Field(max_length=1000)

    mime_type: str | None = Field(default=None, max_length=100)
    file_size_bytes: int | None = Field(default=None, ge=0)
    checksum_sha256: str | None = Field(default=None, max_length=64)

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
        ),
    )
