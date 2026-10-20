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
import hashlib
import time
from datetime import datetime, timezone

from app.model.experiment import Experiment
from app.model.model_run import ModelRun
from app.model.metric import ExperimentMetric
from app.model.artifact import Artifact
from app.services.model_persistence_service import save_model

from app.services.dataset_profiling_service import profile_dataset
from pathlib import Path

import pandas as pd

from app.schemas.dataset import TrainingRequest
from app.services.training_service import train_model
from app.model.experiment import Experiment
from app.model.model_run import ModelRun
from app.model.artifact import Artifact
from app.model.prediction import Prediction
from app.schemas.dataset import PredictionRequest
from app.services.model_persistence_service import load_model
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



@router.post("/{dataset_id}/train")
def train_dataset_model(
    dataset_id: int,
    request: TrainingRequest,
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
        raise HTTPException(status_code=404, detail="Dataset not found.")

    file_path = Path(dataset.file_path)
    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Dataset file not found on the server.",
        )

    try:
        df = pd.read_csv(file_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the dataset CSV.",
        ) from exc

    experiment = Experiment(
        user_id=current_user.user_id,
        dataset_id=dataset.dataset_id,
        name=f"{dataset.name} - {request.algorithm}",
        task_type=request.task_type,
        target_column=request.target_column,
        status="running",
        config={
            "feature_columns": request.feature_columns,
            "test_size": request.test_size,
            "random_state": 42,
        },
    )
    session.add(experiment)
    session.commit()
    session.refresh(experiment)

    run = ModelRun(
        experiment_id=experiment.experiment_id,
        algorithm=request.algorithm,
        hyperparameters={
            "test_size": request.test_size,
            "random_state": 42,
        },
        status="running",
        started_at=datetime.now(timezone.utc),
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    start_time = time.perf_counter()

    try:
        result = train_model(
            df=df,
            feature_columns=request.feature_columns,
            target_column=request.target_column,
            task_type=request.task_type,
            algorithm=request.algorithm,
            test_size=request.test_size,
        )

        artifact_data = save_model(result["pipeline"], run.run_id)

        artifact = Artifact(
            experiment_id=experiment.experiment_id,
            run_id=run.run_id,
            name=artifact_data["name"],
            artifact_type=artifact_data["artifact_type"],
            file_path=artifact_data["file_path"],
            mime_type=artifact_data["mime_type"],
            file_size_bytes=artifact_data["file_size_bytes"],
            checksum_sha256=artifact_data["checksum_sha256"],
        )
        session.add(artifact)

        for metric_name, metric_value in result["metrics"].items():
            session.add(
                ExperimentMetric(
                    run_id=run.run_id,
                    metric_name=metric_name,
                    metric_value=float(metric_value),
                    dataset_split="test",
                )
            )

        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc)
        run.duration_seconds = time.perf_counter() - start_time
        experiment.status = "completed"

        session.add(run)
        session.add(experiment)
        session.commit()
        session.refresh(artifact)

        return {
            "experiment_id": experiment.experiment_id,
            "run_id": run.run_id,
            "artifact_id": artifact.artifact_id,
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.name,
            "task_type": result["task_type"],
            "algorithm": result["algorithm"],
            "feature_columns": result["feature_columns"],
            "target_column": result["target_column"],
            "train_rows": result["train_rows"],
            "test_rows": result["test_rows"],
            "metrics": result["metrics"],
            "model_artifact": {
                "name": artifact.name,
                "file_size_bytes": artifact.file_size_bytes,
                "checksum_sha256": artifact.checksum_sha256,
            },
        }

    except Exception as exc:
        session.rollback()

        run.status = "failed"
        run.completed_at = datetime.now(timezone.utc)
        run.duration_seconds = time.perf_counter() - start_time
        run.error_message = str(exc)[:2000]
        experiment.status = "failed"

        session.add(run)
        session.add(experiment)
        session.commit()

        if isinstance(exc, ValueError):
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        raise HTTPException(
            status_code=500,
            detail="Model training or persistence failed.",
        ) from exc



@router.post("/runs/{run_id}/predict")
def predict_with_model(
    run_id: int,
    request: PredictionRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> dict:
    run = session.exec(
        select(ModelRun)
        .join(Experiment, ModelRun.experiment_id == Experiment.experiment_id)
        .where(
            ModelRun.run_id == run_id,
            Experiment.user_id == current_user.user_id,
            ModelRun.status == "completed",
        )
    ).first()

    if run is None:
        raise HTTPException(
            status_code=404,
            detail="Completed model run not found.",
        )

    artifact = session.exec(
        select(Artifact).where(
            Artifact.run_id == run_id,
            Artifact.experiment_id == run.experiment_id,
            Artifact.artifact_type == "trained_model",
        )
    ).first()

    if artifact is None:
        raise HTTPException(
            status_code=404,
            detail="Trained model artifact not found.",
        )

    try:
        pipeline = load_model(
            artifact.file_path,
            artifact.checksum_sha256,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Saved model file not found.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail="Model artifact verification failed.",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to read the model artifact.",
        ) from exc

    expected_features = list(pipeline.feature_names_in_)
    supplied_features = list(request.input_data.keys())

    if set(supplied_features) != set(expected_features):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Input features do not match the model's expected features.",
                "expected_features": expected_features,
                "received_features": supplied_features,
            },
        )

    try:
        input_df = pd.DataFrame(
            [request.input_data],
            columns=expected_features,
        )
        prediction_value = float(pipeline.predict(input_df)[0])
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid input values for this model.",
        ) from exc

    prediction = Prediction(
        run_id=run_id,
        input_data=request.input_data,
        output_data={
            "target_column": session.get(
                Experiment, run.experiment_id
            ).target_column,
            "predicted_value": prediction_value,
        },
        status="success",
    )

    session.add(prediction)
    session.commit()
    session.refresh(prediction)

    return {
        "prediction_id": prediction.prediction_id,
        "run_id": run_id,
        "input_data": prediction.input_data,
        "output_data": prediction.output_data,
        "status": prediction.status,
        "created_at": prediction.created_at,
    }



@router.get("/runs/{run_id}/predictions")
def get_prediction_history(
    run_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> dict:
    run = session.exec(
        select(ModelRun)
        .join(
            Experiment,
            ModelRun.experiment_id == Experiment.experiment_id,
        )
        .where(
            ModelRun.run_id == run_id,
            Experiment.user_id == current_user.user_id,
        )
    ).first()

    if run is None:
        raise HTTPException(
            status_code=404,
            detail="Model run not found.",
        )

    predictions = session.exec(
        select(Prediction)
        .where(Prediction.run_id == run_id)
        .order_by(Prediction.created_at.desc())
    ).all()

    return {
        "run_id": run_id,
        "total_predictions": len(predictions),
        "predictions": [
            {
                "prediction_id": item.prediction_id,
                "input_data": item.input_data,
                "output_data": item.output_data,
                "status": item.status,
                "error_message": item.error_message,
                "created_at": item.created_at,
            }
            for item in predictions
        ],
    }
