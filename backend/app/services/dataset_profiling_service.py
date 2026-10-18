import pandas as pd
from fastapi import HTTPException
from pathlib import Path

from app.model.dataset import Dataset


def profile_dataset(dataset: Dataset) -> dict:
    file_path = Path(dataset.file_path)

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Dataset file was not found.",
        )

    try:
        df = pd.read_csv(file_path)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError):
        raise HTTPException(
            status_code=422,
            detail="Unable to read this CSV for profiling.",
        )

    missing = df.isna().sum()

    columns = []

    for column in df.columns:
        series = df[column]
        columns.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "missing_count": int(missing[column]),
                "missing_percentage": round(
                    float(missing[column] / len(df) * 100), 2
                ) if len(df) else 0.0,
                "unique_count": int(series.nunique(dropna=True)),
            }
        )

    numeric_columns = df.select_dtypes(include="number").columns
    categorical_columns = df.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns

    numeric_summary = (
        df[numeric_columns].describe().to_dict()
        if len(numeric_columns) > 0
        else {}
    )

    categorical_summary = (
        df[categorical_columns].describe().to_dict()
        if len(categorical_columns) > 0
        else {}
    )
    return {
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": columns,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "preview": df.head(5).fillna("").to_dict(orient="records"),
    }