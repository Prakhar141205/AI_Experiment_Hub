
import pandas as pd
import pytest

from app.services.training_service import train_model


df = pd.DataFrame({
    "hours_studied": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        2, 4, 6, 8, 10, 3, 5, 7, 9, 1,
    ],
    "attendance": [
        50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
        52, 62, 72, 82, 92, 58, 68, 78, 88, 48,
    ],
    "study_method": [
        "self", "group", "self", "group", "self",
        "group", "self", "group", "self", "group",
        "self", "group", "self", "group", "self",
        "group", "self", "group", "self", "group",
    ],
    "marks": [
        35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
        38, 48, 58, 68, 78, 42, 52, 62, 72, 33,
    ],
})

classification_df = pd.DataFrame({
    "hours_studied": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        2, 4, 6, 8, 10, 3, 5, 7, 9, 1,
    ],
    "attendance": [
        50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
        52, 62, 72, 82, 92, 58, 68, 78, 88, 48,
    ],
    "study_method": [
        "self", "group", "self", "group", "self",
        "group", "self", "group", "self", "group",
        "self", "group", "self", "group", "self",
        "group", "self", "group", "self", "group",
    ],
    "result": [
        "fail", "fail", "fail", "fail", "pass",
        "pass", "pass", "pass", "pass", "pass",
        "fail", "fail", "pass", "pass", "pass",
        "fail", "pass", "pass", "pass", "fail",
    ],
})

FEATURE_COLUMNS = [
    "hours_studied",
    "attendance",
    "study_method",
]

REGRESSION_ALGORITHMS = [
    "linear_regression",
    "ridge",
    "lasso",
    "svr",
    "decision_tree",
    "random_forest",
]

CLASSIFICATION_ALGORITHMS = [
    "logistic_regression",
    "svm",
    "decision_tree",
    "random_forest",
    "knn",
]


@pytest.mark.parametrize("algorithm", REGRESSION_ALGORITHMS)
def test_regression_algorithms(algorithm):
    result = train_model(
        df=df,
        feature_columns=FEATURE_COLUMNS,
        target_column="marks",
        task_type="regression",
        algorithm=algorithm,
        test_size=0.2,
        random_state=42,
    )

    assert result["task_type"] == "regression"
    assert result["algorithm"] == algorithm
    assert result["train_rows"] > 0
    assert result["test_rows"] > 0
    assert result["train_rows"] + result["test_rows"] == len(df)
    assert result["pipeline"] is not None

    for metric in ("mae", "mse", "rmse", "r2"):
        assert metric in result["metrics"]


@pytest.mark.parametrize("algorithm", CLASSIFICATION_ALGORITHMS)
def test_classification_algorithms(algorithm):
    result = train_model(
        df=classification_df,
        feature_columns=FEATURE_COLUMNS,
        target_column="result",
        task_type="classification",
        algorithm=algorithm,
        test_size=0.2,
        random_state=42,
    )

    assert result["task_type"] == "classification"
    assert result["algorithm"] == algorithm
    assert result["train_rows"] > 0
    assert result["test_rows"] > 0
    assert (
        result["train_rows"] + result["test_rows"]
        == len(classification_df)
    )
    assert result["pipeline"] is not None

    for metric in ("accuracy", "precision_weighted",
                   "recall_weighted", "f1_weighted"):
        assert metric in result["metrics"]


def test_random_state_and_hyperparameters_are_returned():
    result = train_model(
        df=df,
        feature_columns=FEATURE_COLUMNS,
        target_column="marks",
        task_type="regression",
        algorithm="random_forest",
        random_state=17,
        hyperparameters={"n_estimators": 20, "max_depth": 4},
    )

    assert result["random_state"] == 17
    assert result["hyperparameters"]["n_estimators"] == 20
    assert result["hyperparameters"]["max_depth"] == 4
