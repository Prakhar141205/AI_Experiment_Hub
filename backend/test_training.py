
import pandas as pd

from app.services.training_service import train_model


# -----------------------------
# Sample data for regression
# -----------------------------
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


# -----------------------------
# Sample data for classification
# -----------------------------
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


# -----------------------------
# Regression test
# -----------------------------
def test_linear_regression():
    result = train_model(
        df=df,
        feature_columns=FEATURE_COLUMNS,
        target_column="marks",
        task_type="regression",
        algorithm="linear_regression",
    )

    assert result["task_type"] == "regression"
    assert result["algorithm"] == "linear_regression"
    assert result["train_rows"] > 0
    assert result["test_rows"] > 0
    assert result["train_rows"] + result["test_rows"] == len(df)

    assert "mae" in result["metrics"]
    assert "rmse" in result["metrics"]
    assert "r2" in result["metrics"]

    assert result["pipeline"] is not None


# -----------------------------
# Classification test
# -----------------------------
def test_logistic_regression():
    result = train_model(
        df=classification_df,
        feature_columns=FEATURE_COLUMNS,
        target_column="result",
        task_type="classification",
        algorithm="logistic_regression",
    )

    assert result["task_type"] == "classification"
    assert result["algorithm"] == "logistic_regression"
    assert result["train_rows"] > 0
    assert result["test_rows"] > 0
    assert (
        result["train_rows"] + result["test_rows"]
        == len(classification_df)
    )

    assert "accuracy" in result["metrics"]
    assert result["pipeline"] is not None
