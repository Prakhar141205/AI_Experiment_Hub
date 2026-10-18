from datetime import datetime

from sqlmodel import SQLModel


class DatasetRead(SQLModel):
    dataset_id: int
    name: str
    original_filename: str
    row_count: int
    column_count: int
    uploaded_at: datetime