from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.services.preprocessing_service import build_preprocessor


# Each task has its own explicit allowlist of estimators. Never accept
# arbitrary Python classes or import paths from an API request.
ALGORITHMS: dict[str, dict[str, type]] = {
    "classification": {
        "logistic_regression": LogisticRegression,
        "svm": SVC,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "knn": KNeighborsClassifier,
    },
    "regression": {
        "linear_regression": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "svr": SVR,
        "decision_tree": DecisionTreeRegressor,
        "random_forest": RandomForestRegressor,
    },
}


def _build_estimator(
    task_type: str,
    algorithm: str,
    hyperparameters: dict[str, Any] | None,
    random_state: int,
):
    """Create an allowlisted estimator and validate its parameters."""
    if task_type not in ALGORITHMS:
        raise ValueError(
            "Unsupported task type. Choose classification or regression."
        )

    if algorithm not in ALGORITHMS[task_type]:
        available = ", ".join(sorted(ALGORITHMS[task_type]))
        raise ValueError(
            f"Algorithm '{algorithm}' is not supported for {task_type}. "
            f"Available algorithms: {available}."
        )

    estimator_class = ALGORITHMS[task_type][algorithm]
    params = dict(hyperparameters or {})

    # Use deterministic defaults where the estimator supports random_state.
    allowed_params = estimator_class().get_params(deep=False)
    if "random_state" in allowed_params:
        params.setdefault("random_state", random_state)
    if algorithm == "logistic_regression":
        params.setdefault("max_iter", 1000)

    unknown_params = sorted(set(params) - set(allowed_params))
    if unknown_params:
        raise ValueError(
            f"Unsupported hyperparameters for '{algorithm}': "
            f"{unknown_params}."
        )

    try:
        return estimator_class(**params)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid hyperparameters for '{algorithm}': {exc}"
        ) from exc


def train_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    task_type: str,
    algorithm: str,
    test_size: float = 0.2,
    random_state: int = 42,
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train an allowlisted tabular ML model and return its pipeline and metrics."""

    if df.empty:
        raise ValueError("Dataset cannot be empty.")
    if not feature_columns:
        raise ValueError("Select at least one input feature column.")
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Duplicate feature columns are not allowed.")
    if target_column in feature_columns:
        raise ValueError("Target column cannot also be an input feature.")

    required_columns = [*feature_columns, target_column]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in dataset: {missing_columns}")
    if df.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")
    if df[target_column].isna().any():
        raise ValueError("Target column contains missing values. Clean it first.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not isinstance(random_state, int):
        raise ValueError("random_state must be an integer.")

    estimator = _build_estimator(
        task_type=task_type,
        algorithm=algorithm,
        hyperparameters=hyperparameters,
        random_state=random_state,
    )

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    if task_type == "classification":
        if y.nunique() < 2:
            raise ValueError("Classification requires at least two target classes.")
        class_counts = y.value_counts()
        stratify = y if class_counts.min() >= 2 else None
    else:
        try:
            y = pd.to_numeric(y, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Regression target must contain numeric values."
            ) from exc
        if y.nunique() < 2:
            raise ValueError("Regression requires at least two distinct target values.")
        stratify = None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError as exc:
        raise ValueError(f"Unable to split dataset for training: {exc}") from exc

    numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    supported_count = len(numeric_columns) + len(categorical_columns)
    if supported_count != X_train.shape[1]:
        unsupported = [
            col for col in X_train.columns
            if col not in numeric_columns and col not in categorical_columns
        ]
        raise ValueError(
            "Some selected features have unsupported data types: "
            f"{unsupported}. Convert them to numeric or categorical values first."
        )

    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    if task_type == "classification":
        metrics: dict[str, float | int | None] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_weighted": float(
                precision_score(
                    y_test, predictions, average="weighted", zero_division=0
                )
            ),
            "recall_weighted": float(
                recall_score(
                    y_test, predictions, average="weighted", zero_division=0
                )
            ),
            "f1_weighted": float(
                f1_score(
                    y_test, predictions, average="weighted", zero_division=0
                )
            ),
        }
    else:
        mse = mean_squared_error(y_test, predictions)
        metrics = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "mse": float(mse),
            "rmse": float(mse ** 0.5),
            "r2": float(r2_score(y_test, predictions)) if len(y_test) >= 2 else None,
        }

    return {
        "task_type": task_type,
        "algorithm": algorithm,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "hyperparameters": estimator.get_params(deep=False),
        "random_state": random_state,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": metrics,
        "pipeline": pipeline,
    }
