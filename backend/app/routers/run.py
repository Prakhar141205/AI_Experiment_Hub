
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from app.core.dependencies import get_current_user
from app.database.database import get_db
from app.model.artifact import Artifact
from app.model.experiment import Experiment
from app.model.metric import ExperimentMetric
from app.model.model_run import ModelRun
from app.model.prediction import Prediction
from app.model.user import User


router = APIRouter(tags=["Model Runs"])


def get_owned_run(
    run_id: int,
    user_id: int,
    session: Session,
) -> ModelRun:
    run = session.exec(
        select(ModelRun)
        .join(
            Experiment,
            ModelRun.experiment_id == Experiment.experiment_id,
        )
        .where(
            ModelRun.run_id == run_id,
            Experiment.user_id == user_id,
        )
    ).first()

    if run is None:
        raise HTTPException(
            status_code=404,
            detail="Model run not found.",
        )

    return run


@router.get("/experiments/{experiment_id}/runs")
def list_experiment_runs(
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

    runs = session.exec(
        select(ModelRun)
        .where(ModelRun.experiment_id == experiment_id)
        .order_by(ModelRun.created_at.desc())
    ).all()

    return {
        "experiment_id": experiment_id,
        "total": len(runs),
        "runs": [
            {
                "run_id": run.run_id,
                "algorithm": run.algorithm,
                "model_version": run.model_version,
                "hyperparameters": run.hyperparameters,
                "status": run.status,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
                "duration_seconds": run.duration_seconds,
                "error_message": (
                    run.error_message if run.status == "failed" else None
                ),
                "created_at": run.created_at,
            }
            for run in runs
        ],
    }


@router.get("/runs/{run_id}")
def get_run_details(
    run_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> dict:
    run = get_owned_run(run_id, current_user.user_id, session)

    return {
        "run_id": run.run_id,
        "experiment_id": run.experiment_id,
        "algorithm": run.algorithm,
        "model_version": run.model_version,
        "hyperparameters": run.hyperparameters,
        "status": run.status,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "duration_seconds": run.duration_seconds,
        "error_message": (
            run.error_message if run.status == "failed" else None
        ),
        "created_at": run.created_at,
    }


@router.get("/runs/{run_id}/metrics")
def get_run_metrics(
    run_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> dict:
    get_owned_run(run_id, current_user.user_id, session)

    metrics = session.exec(
        select(ExperimentMetric)
        .where(ExperimentMetric.run_id == run_id)
        .order_by(ExperimentMetric.metric_id)
    ).all()

    return {
        "run_id": run_id,
        "total": len(metrics),
        "metrics": [
            {
                "metric_id": metric.metric_id,
                "metric_name": metric.metric_name,
                "metric_value": metric.metric_value,
                "dataset_split": metric.dataset_split,
                "created_at": metric.created_at,
            }
            for metric in metrics
        ],
    }


@router.get("/runs/{run_id}/artifacts")
def list_run_artifacts(
    run_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> dict:
    run = get_owned_run(run_id, current_user.user_id, session)

    artifacts = session.exec(
        select(Artifact)
        .where(
            Artifact.run_id == run_id,
            Artifact.experiment_id == run.experiment_id,
        )
        .order_by(Artifact.created_at.desc())
    ).all()

    return {
        "run_id": run_id,
        "total": len(artifacts),
        "artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "name": artifact.name,
                "artifact_type": artifact.artifact_type,
                "mime_type": artifact.mime_type,
                "file_size_bytes": artifact.file_size_bytes,
                "checksum_sha256": artifact.checksum_sha256,
                "created_at": artifact.created_at,
            }
            for artifact in artifacts
        ],
    }


@router.get("/artifacts/{artifact_id}/download")
def download_artifact(
    artifact_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> FileResponse:
    artifact = session.exec(
        select(Artifact)
        .join(
            Experiment,
            Artifact.experiment_id == Experiment.experiment_id,
        )
        .where(
            Artifact.artifact_id == artifact_id,
            Experiment.user_id == current_user.user_id,
        )
    ).first()

    if artifact is None:
        raise HTTPException(
            status_code=404,
            detail="Artifact not found.",
        )

    file_path = Path(artifact.file_path)

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Artifact file not found on the server.",
        )

    return FileResponse(
        path=file_path,
        filename=artifact.name,
        media_type=artifact.mime_type or "application/octet-stream",
    )


@router.get("/predictions/{prediction_id}")
def get_prediction_details(
    prediction_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> dict:
    prediction = session.exec(
        select(Prediction)
        .join(
            ModelRun,
            Prediction.run_id == ModelRun.run_id,
        )
        .join(
            Experiment,
            ModelRun.experiment_id == Experiment.experiment_id,
        )
        .where(
            Prediction.prediction_id == prediction_id,
            Experiment.user_id == current_user.user_id,
        )
    ).first()

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found.",
        )

    return {
        "prediction_id": prediction.prediction_id,
        "run_id": prediction.run_id,
        "input_data": prediction.input_data,
        "output_data": prediction.output_data,
        "status": prediction.status,
        "error_message": prediction.error_message,
        "created_at": prediction.created_at,
    }
