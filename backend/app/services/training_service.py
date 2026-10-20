
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.services.preprocessing_service import build_preprocessor


ALGORITHMS = {
    "classification": {
        "logistic_regression": LogisticRegression(max_iter=1000),
    },
    "regression": {
        "linear_regression": LinearRegression(),
    },
}


def train_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    task_type: str,
    algorithm: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a model using selected features and return evaluation metrics."""

    if df.empty:
        raise ValueError("Dataset cannot be empty.")

    if not feature_columns:
        raise ValueError("Select at least one input feature column.")

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Duplicate feature columns are not allowed.")

    if target_column in feature_columns:
        raise ValueError(
            "Target column cannot also be an input feature."
        )

    missing_columns = [
        column
        for column in [*feature_columns, target_column]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Columns not found in dataset: {missing_columns}"
        )

    if df[target_column].isna().any():
        raise ValueError(
            "Target column contains missing values. Clean it first."
        )

    if task_type not in ALGORITHMS:
        raise ValueError("Unsupported task type.")

    if algorithm not in ALGORITHMS[task_type]:
        raise ValueError(
            f"Algorithm '{algorithm}' is not supported for {task_type}."
        )

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    if task_type == "classification" and y.nunique() < 2:
        raise ValueError(
            "Classification requires at least two target classes."
        )

    stratify = None

    if task_type == "classification":
        class_counts = y.value_counts()

        if class_counts.min() >= 2:
            stratify = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError as exc:
        raise ValueError(
            f"Unable to split dataset for training: {exc}"
        ) from exc

    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    supported_count = len(numeric_columns) + len(categorical_columns)

    if supported_count != X_train.shape[1]:
        raise ValueError(
            "Some selected features have unsupported data types. "
            "Convert them to numeric or categorical values first."
        )

    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    model = clone(ALGORITHMS[task_type][algorithm])

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    if task_type == "classification":
        metrics = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "f1_weighted": float(
                f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
        }
    else:
        mse = mean_squared_error(y_test, predictions)

        metrics = {
            "mae": float(
                mean_absolute_error(y_test, predictions)
            ),
            "rmse": float(mse ** 0.5),
            "r2": (
                float(r2_score(y_test, predictions))
                if len(y_test) >= 2
                else None
            ),
        }

    return {
        "task_type": task_type,
        "algorithm": algorithm,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": metrics,
        "pipeline": pipeline,
    }
