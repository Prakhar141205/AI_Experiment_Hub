import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the backend package importable when pytest runs from the project root.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.training_service import ALGORITHMS, train_model


# ---------------------------------------------------------
# Test datasets
# ---------------------------------------------------------

@pytest.fixture
def classification_data():
    rng = np.random.default_rng(42)
    n_samples = 150

    feature_1 = rng.normal(size=n_samples)
    feature_2 = rng.normal(size=n_samples)
    feature_3 = rng.uniform(0, 10, size=n_samples)

    score = 2 * feature_1 - feature_2 + 0.2 * feature_3
    target = np.where(
        score > np.median(score),
        "Class_A",
        "Class_B",
    )

    return pd.DataFrame({
        "feature_1": feature_1,
        "feature_2": feature_2,
        "feature_3": feature_3,
        "target": target,
    })


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    n_samples = 150

    feature_1 = rng.normal(size=n_samples)
    feature_2 = rng.normal(size=n_samples)
    feature_3 = rng.uniform(0, 10, size=n_samples)
    noise = rng.normal(0, 0.1, size=n_samples)

    target = (
        3 * feature_1
        - 2 * feature_2
        + 0.5 * feature_3
        + noise
    )

    return pd.DataFrame({
        "feature_1": feature_1,
        "feature_2": feature_2,
        "feature_3": feature_3,
        "target": target,
    })


# ---------------------------------------------------------
# Supported algorithms
# ---------------------------------------------------------

CLASSIFICATION_ALGORITHMS = [
    "logistic_regression",
    "svm",
    "decision_tree",
    "random_forest",
    "knn",
]

REGRESSION_ALGORITHMS = [
    "linear_regression",
    "ridge",
    "lasso",
    "svr",
    "decision_tree",
    "random_forest",
]


# ---------------------------------------------------------
# Classification tests
# ---------------------------------------------------------

@pytest.mark.parametrize("algorithm", CLASSIFICATION_ALGORITHMS)
def test_classification_algorithms(classification_data, algorithm):
    result = train_model(
        df=classification_data,
        feature_columns=["feature_1", "feature_2", "feature_3"],
        target_column="target",
        task_type="classification",
        algorithm=algorithm,
        test_size=0.2,
        random_state=42,
    )

    assert isinstance(result, dict)
    assert result["pipeline"] is not None
    assert result["task_type"] == "classification"
    assert result["algorithm"] == algorithm
    assert result["random_state"] == 42

    metrics = result["metrics"]

    for metric in (
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ):
        assert metric in metrics
        assert np.isfinite(metrics[metric])
        assert 0 <= metrics[metric] <= 1

    assert result["train_rows"] == 120
    assert result["test_rows"] == 30


# ---------------------------------------------------------
# Regression tests
# ---------------------------------------------------------

@pytest.mark.parametrize("algorithm", REGRESSION_ALGORITHMS)
def test_regression_algorithms(regression_data, algorithm):
    result = train_model(
        df=regression_data,
        feature_columns=["feature_1", "feature_2", "feature_3"],
        target_column="target",
        task_type="regression",
        algorithm=algorithm,
        test_size=0.2,
        random_state=42,
    )

    assert isinstance(result, dict)
    assert result["pipeline"] is not None
    assert result["task_type"] == "regression"
    assert result["algorithm"] == algorithm
    assert result["random_state"] == 42

    metrics = result["metrics"]

    for metric in ("mae", "mse", "rmse", "r2"):
        assert metric in metrics

        # R² can be negative; other regression error metrics cannot.
        if metric == "r2":
            assert metrics[metric] is None or np.isfinite(metrics[metric])
        else:
            assert np.isfinite(metrics[metric])
            assert metrics[metric] >= 0

    assert result["train_rows"] == 120
    assert result["test_rows"] == 30


# ---------------------------------------------------------
# Hyperparameter and configuration tests
# ---------------------------------------------------------

def test_random_state_and_hyperparameters(regression_data):
    supplied_hyperparameters = {
        "n_estimators": 25,
        "max_depth": 5,
    }

    result = train_model(
        df=regression_data,
        feature_columns=["feature_1", "feature_2", "feature_3"],
        target_column="target",
        task_type="regression",
        algorithm="random_forest",
        test_size=0.25,
        random_state=123,
        hyperparameters=supplied_hyperparameters,
    )

    assert result["algorithm"] == "random_forest"
    assert result["task_type"] == "regression"
    assert result["random_state"] == 123
    assert result["train_rows"] == 112
    assert result["test_rows"] == 38

    # The service returns the estimator's effective parameters.
    returned_parameters = result["hyperparameters"]
    assert returned_parameters["n_estimators"] == 25
    assert returned_parameters["max_depth"] == 5


# ---------------------------------------------------------
# Algorithm registry tests
# ---------------------------------------------------------

def test_classification_algorithm_registry():
    assert set(CLASSIFICATION_ALGORITHMS).issubset(
        set(ALGORITHMS["classification"])
    )


def test_regression_algorithm_registry():
    assert set(REGRESSION_ALGORITHMS).issubset(
        set(ALGORITHMS["regression"])
    )


# ---------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------

def test_empty_dataframe_is_rejected():
    with pytest.raises(ValueError, match="Dataset cannot be empty"):
        train_model(
            df=pd.DataFrame(),
            feature_columns=["feature_1"],
            target_column="target",
            task_type="regression",
            algorithm="linear_regression",
        )


def test_classification_with_one_class_is_rejected():
    df = pd.DataFrame({
        "feature_1": [1, 2, 3, 4],
        "target": ["A", "A", "A", "A"],
    })

    with pytest.raises(
        ValueError,
        match="at least two target classes",
    ):
        train_model(
            df=df,
            feature_columns=["feature_1"],
            target_column="target",
            task_type="classification",
            algorithm="logistic_regression",
        )


def test_regression_with_non_numeric_target_is_rejected():
    df = pd.DataFrame({
        "feature_1": [1, 2, 3, 4, 5, 6],
        "target": ["low", "high", "low", "high", "low", "high"],
    })

    with pytest.raises(
        ValueError,
        match="Regression target must contain numeric values",
    ):
        train_model(
            df=df,
            feature_columns=["feature_1"],
            target_column="target",
            task_type="regression",
            algorithm="linear_regression",
        )