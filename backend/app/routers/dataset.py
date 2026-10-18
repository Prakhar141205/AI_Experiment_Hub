from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlmodel import Session

from app.core.dependencies import get_current_user
from app.database.database import get_db
from app.model.dataset import Dataset
from app.model.user import User
from app.schemas.dataset import DatasetRead
from app.services.dataset_service import upload_dataset
from fastapi import HTTPException
from sqlmodel import select

from app.services.dataset_profiling_service import profile_dataset


router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.post(
    "/upload",
    response_model=DatasetRead,
    status_code=status.HTTP_201_CREATED,
)
def create_dataset(
    name: Annotated[str, Form(min_length=1, max_length=100)],
    file: Annotated[UploadFile, File()],
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> Dataset:
    return upload_dataset(
        session=session,
        user_id=current_user.user_id,
        name=name,
        file=file,
    )


@router.get("/{dataset_id}/profile")
def get_dataset_profile(
    dataset_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> dict:
    dataset = session.exec(
        select(Dataset).where(
            Dataset.dataset_id == dataset_id,
            Dataset.user_id == current_user.user_id,
        )
    ).first()

    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    return profile_dataset(dataset)