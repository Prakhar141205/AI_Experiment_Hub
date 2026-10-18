import csv
import io
import os
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session

from app.model.dataset import Dataset

UPLOAD_DIR = Path("uploads/datasets")
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
CHUNK_SIZE = 1024 * 1024  # 1 MB


def upload_dataset(
    session: Session,
    user_id: int,
    name: str,
    file: UploadFile,
) -> Dataset:
    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are supported.",
        )

    stored_path: Path | None = None

    try:
        content = file.file.read(MAX_FILE_SIZE + 1)

        if not content:
            raise HTTPException(400, "The uploaded file is empty.")

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(413, "File exceeds the 20 MB upload limit.")

        text = content.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text, newline=""))

        try:
            header = next(reader)
        except StopIteration:
            raise HTTPException(400, "The CSV has no header row.")

        if not header or not any(column.strip() for column in header):
            raise HTTPException(400, "The CSV header is invalid.")

        if any(not column.strip() for column in header):
            raise HTTPException(400, "Column names cannot be empty.")

        if len(set(header)) != len(header):
            raise HTTPException(400, "Duplicate column names are not supported.")

        row_count = 0

        for row in reader:
            if len(row) != len(header):
                raise HTTPException(
                    400,
                    f"Invalid CSV structure near data row {row_count + 1}.",
                )
            row_count += 1

        # Store using a generated filename, never the user-supplied filename.
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        stored_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.csv"

        with stored_path.open("xb") as destination:
            destination.write(content)

        dataset = Dataset(
            user_id=user_id,
            name=name.strip(),
            original_filename=Path(file.filename).name,
            file_path=str(stored_path),
            row_count=row_count,
            column_count=len(header),
        )

        session.add(dataset)
        session.commit()
        session.refresh(dataset)

        return dataset

    except HTTPException:
        if stored_path is not None:
            stored_path.unlink(missing_ok=True)
        raise

    except (UnicodeDecodeError, csv.Error):
        if stored_path is not None:
            stored_path.unlink(missing_ok=True)
        raise HTTPException(400, "The file is not a valid UTF-8 CSV.")

    except (OSError, SQLAlchemyError):
        session.rollback()
        if stored_path is not None:
            stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Dataset upload failed. Please try again.",
        )