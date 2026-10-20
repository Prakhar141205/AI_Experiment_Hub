from datetime import datetime
from typing import Any, Literal

from sqlmodel import Field, SQLModel


class DatasetRead(SQLModel):
    dataset_id: int
    name: str
    original_filename: str
    row_count: int
    column_count: int
    uploaded_at: datetime


class TrainingRequest(SQLModel):
    feature_columns: list[str] = Field(min_length=1)
    target_column: str = Field(min_length=1, max_length=255)
    task_type: Literal["classification", "regression"]
    algorithm: Literal[
        "logistic_regression",
        "svm",
        "decision_tree",
        "random_forest",
        "knn",
        "linear_regression",
        "ridge",
        "lasso",
        "svr",
    ]
    test_size: float = Field(default=0.2, gt=0, lt=1)
    random_state: int = Field(default=42, ge=0)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class PredictionRequest(SQLModel):
    input_data: dict[str, Any] = Field(min_length=1)
