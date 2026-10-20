from datetime import datetime

from sqlmodel import SQLModel, Field
from typing import Any


class DatasetRead(SQLModel):
    dataset_id: int
    name: str
    original_filename: str
    row_count: int
    column_count: int
    uploaded_at: datetime

from typing import Literal

class TrainingRequest(SQLModel):
    feature_columns: list[str] = Field(min_length=1)
    target_column: str = Field(min_length=1, max_length=100)
    task_type: Literal["classification", "regression"]
    algorithm: Literal["logistic_regression", "linear_regression"]
    test_size: float = Field(default=0.2, gt=0, lt=1)

class PredictionRequest(SQLModel):
    input_data: dict[str, Any] = Field(min_length=1)