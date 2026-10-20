
from pathlib import Path

import pandas as pd


def profile_dataset(file_path: str | Path) -> dict:
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    missing_values = df.isnull().sum()

    return {
        "filename": file_path.name,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "column_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_values": {
            column: int(count)
            for column, count in missing_values.items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_bytes": int(
            df.memory_usage(deep=True).sum()
        ),
        "numeric_summary": df.describe(
            include="number"
        ).to_dict(),
    }