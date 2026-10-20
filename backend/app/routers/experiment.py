
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from app.core.dependencies import get_current_user
from app.database.database import get_db
from app.model.experiment import Experiment
from app.model.user import User

from fastapi import HTTPException

router = APIRouter(prefix="/experiments", tags=["Experiments"])


@router.get("/")
def get_experiments(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> dict:
    experiments = session.exec(
        select(Experiment)
        .where(Experiment.user_id == current_user.user_id)
        .order_by(Experiment.created_at.desc())
    ).all()

    return {
        "total": len(experiments),
        "experiments": [
            {
                "experiment_id": experiment.experiment_id,
                "dataset_id": experiment.dataset_id,
                "name": experiment.name,
                "description": experiment.description,
                "task_type": experiment.task_type,
                "target_column": experiment.target_column,
                "status": experiment.status,
                "config": experiment.config,
                "created_at": experiment.created_at,
                "updated_at": experiment.updated_at,
            }
            for experiment in experiments
        ],
    }



@router.get("/{experiment_id}")
def get_experiment(
    experiment_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> dict:
    experiment = session.exec(
        select(Experiment).where(
            Experiment.experiment_id == experiment_id,
            Experiment.user_id == current_user.user_id,
        )
    ).first()

    if experiment is None:

        raise HTTPException(
            status_code=404,
            detail="Experiment not found.",
        )

    return {
        "experiment_id": experiment.experiment_id,
        "dataset_id": experiment.dataset_id,
        "name": experiment.name,
        "description": experiment.description,
        "task_type": experiment.task_type,
        "target_column": experiment.target_column,
        "status": experiment.status,
        "config": experiment.config,
        "created_at": experiment.created_at,
        "updated_at": experiment.updated_at,
    }
