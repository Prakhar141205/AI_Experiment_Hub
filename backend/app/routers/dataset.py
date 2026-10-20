
import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from sqlmodel import Session, select
from app.services.training_service import ALGORITHMS, train_model
from app.core.dependencies import get_current_user
from app.database.database import get_db
from app.model.artifact import Artifact
from app.model.dataset import Dataset
from app.model.experiment import Experiment
from app.model.metric import ExperimentMetric
from app.model.model_run import ModelRun
from app.model.prediction import Prediction
from app.model.user import User
from app.schemas.dataset import (
    DatasetRead,
    PredictionRequest,
    TrainingRequest,
    ModelComparisonRequest,
)
from app.services.dataset_profiling_service import profile_dataset
from app.services.dataset_service import upload_dataset
from app.services.model_persistence_service import (
    load_model,
    save_model,
)
from app.services.training_service import train_model


router = APIRouter(prefix="/datasets", tags=["Datasets"])

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# 1. Upload a dataset
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 2. List the current user's datasets
# ---------------------------------------------------------

@router.get("/")
def get_datasets(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> dict:
    datasets = session.exec(
        select(Dataset)
        .where(Dataset.user_id == current_user.user_id)
        .order_by(Dataset.uploaded_at.desc())
    ).all()

    return {
        "total": len(datasets),
        "datasets": [
            {
                "dataset_id": dataset.dataset_id,
                "name": dataset.name,
                "original_filename": dataset.original_filename,
                "row_count": dataset.row_count,
                "column_count": dataset.column_count,
                "uploaded_at": dataset.uploaded_at,
            }
            for dataset in datasets
        ],
    }


# ---------------------------------------------------------
# 3. Get dataset details
# ---------------------------------------------------------

@router.get("/{dataset_id}")
def get_dataset_details(
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found.",
        )

    return {
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "original_filename": dataset.original_filename,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "uploaded_at": dataset.uploaded_at,
    }


# ---------------------------------------------------------
# 4. Delete a dataset
# ---------------------------------------------------------

@router.delete(
    "/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_dataset(
    dataset_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> Response:
    dataset = session.exec(
        select(Dataset).where(
            Dataset.dataset_id == dataset_id,
            Dataset.user_id == current_user.user_id,
        )
    ).first()

    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found.",
        )

    existing_experiment = session.exec(
        select(Experiment).where(
            Experiment.dataset_id == dataset_id
        )
    ).first()

    if existing_experiment is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Cannot delete this dataset because it has associated "
                "experiments. Delete or archive those experiments first."
            ),
        )

    file_path = Path(dataset.file_path)

    try:
        session.delete(dataset)
        session.commit()
    except Exception:
        session.rollback()
        raise

    # Database deletion has succeeded. If file removal fails,
    # log the problem so it can be investigated or retried.
    try:
        if file_path.is_file():
            file_path.unlink()
    except OSError:
        logger.exception(
            "Failed to remove dataset file for dataset_id=%s",
            dataset_id,
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------
# 5. Preview dataset rows
# ---------------------------------------------------------

@router.get("/{dataset_id}/preview")
def get_dataset_preview(
    dataset_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
    limit: int = 10,
) -> dict:
    if not 1 <= limit <= 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Preview limit must be between 1 and 100.",
        )

    dataset = session.exec(
        select(Dataset).where(
            Dataset.dataset_id == dataset_id,
            Dataset.user_id == current_user.user_id,
        )
    ).first()

    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found.",
        )

    file_path = Path(dataset.file_path)

    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on the server.",
        )

    try:
        df = pd.read_csv(file_path, nrows=limit)
    except (
        OSError,
        ValueError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unable to read the dataset CSV. "
                "Check that the file is a valid CSV."
            ),
        ) from exc

    # Convert missing values to Python None for JSON serialization.
    rows = (
        df.astype(object)
        .where(pd.notna(df), None)
        .to_dict(orient="records")
    )

    return {
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "columns": df.columns.tolist(),
        "rows": rows,
        "preview_rows": len(df),
        "limit": limit,
    }


# ---------------------------------------------------------
# 6. Profile a dataset
# ---------------------------------------------------------

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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found.",
        )

    file_path = Path(dataset.file_path)

    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on the server.",
        )

    return profile_dataset(file_path)


# ---------------------------------------------------------
# 7. Train a model on a dataset
# ---------------------------------------------------------

@router.post(
    "/{dataset_id}/train",
    status_code=status.HTTP_201_CREATED,
)
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found.",
        )

    file_path = Path(dataset.file_path)

    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on the server.",
        )

    try:
        df = pd.read_csv(file_path)
    except (
        OSError,
        ValueError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to read the dataset CSV.",
        ) from exc

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot train a model on an empty dataset.",
        )

    if len(df.columns) != len(set(df.columns)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset contains duplicate column names.",
        )

    required_columns = [
        *request.feature_columns,
        request.target_column,
    ]

    if len(request.feature_columns) != len(set(request.feature_columns)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Feature columns must be unique.",
        )

    if request.target_column in request.feature_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The target column cannot also be a feature.",
        )

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "One or more requested columns do not exist.",
                "missing_columns": missing_columns,
            },
        )

    if df[request.target_column].isna().any():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The target column contains missing values.",
        )

    if df[request.target_column].nunique() < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The target column must contain at least two distinct values.",
        )

    experiment = Experiment(
        user_id=current_user.user_id,
        dataset_id=dataset_id,
        name=f"{dataset.name} - {request.algorithm}",
        description="Experiment created through the dataset training endpoint.",
        task_type=request.task_type,
        target_column=request.target_column,
        status="running",
        config={
            "feature_columns": request.feature_columns,
            "test_size": request.test_size,
            "random_state": request.random_state,
            "algorithm": request.algorithm,
            "hyperparameters": request.hyperparameters,
        },
    )

    session.add(experiment)
    session.flush()

    run = ModelRun(
        experiment_id=experiment.experiment_id,
        algorithm=request.algorithm,
        status="running",
        hyperparameters={
            "test_size": request.test_size,
            "random_state": request.random_state,
            **request.hyperparameters,
        },
    )

    session.add(run)
    session.flush()

    started_at = time.perf_counter()
    started_datetime = datetime.now(timezone.utc)

    try:
        training_result = train_model(
            df=df,
            feature_columns=request.feature_columns,
            target_column=request.target_column,
            task_type=request.task_type,
            algorithm=request.algorithm,
            test_size=request.test_size,
            random_state=request.random_state,
            hyperparameters=request.hyperparameters,
        )

        pipeline = training_result["pipeline"]
        metrics = training_result["metrics"]

        model_info = save_model(
            pipeline=pipeline,
            run_id=run.run_id,
        )

        run.model_version = model_info.get("model_version", "1")
        run.status = "completed"
        run.started_at = started_datetime
        run.completed_at = datetime.now(timezone.utc)
        run.duration_seconds = time.perf_counter() - started_at

        experiment.status = "completed"
        experiment.updated_at = datetime.now(timezone.utc)

        artifact = Artifact(
            experiment_id=experiment.experiment_id,
            run_id=run.run_id,
            name=model_info["name"],
            artifact_type="model",
            file_path=model_info["file_path"],
            mime_type="application/octet-stream",
            file_size_bytes=model_info["file_size_bytes"],
            checksum_sha256=model_info["checksum_sha256"],
        )
        session.add(artifact)

        for metric_name, metric_value in metrics.items():
            session.add(
                ExperimentMetric(
                    run_id=run.run_id,
                    metric_name=metric_name,
                    metric_value=float(metric_value),
                    dataset_split="test",
                )
            )

        session.commit()
        session.refresh(experiment)
        session.refresh(run)
        session.refresh(artifact)

        return {
            "message": "Model trained successfully.",
            "experiment_id": experiment.experiment_id,
            "run_id": run.run_id,
            "artifact_id": artifact.artifact_id,
            "task_type": experiment.task_type,
            "algorithm": run.algorithm,
            "status": run.status,
            "metrics": metrics,
            "model": {
                "filename": model_info["name"],
                "file_size_bytes": model_info["file_size_bytes"],
                "checksum_sha256": model_info["checksum_sha256"],
            },
            "duration_seconds": run.duration_seconds,
        }

    except HTTPException:
        session.rollback()
        raise
    except Exception as exc:
        session.rollback()

        logger.exception(
            "Training failed for dataset_id=%s",
            dataset_id,
        )

        # Best-effort persistence of a failed run. Rollback above
        # may have expired or detached these objects, so re-query.
        failed_run = session.get(ModelRun, run.run_id)
        failed_experiment = session.get(
            Experiment,
            experiment.experiment_id,
        )

        if failed_run is not None:
            failed_run.status = "failed"
            failed_run.error_message = str(exc)[:1000]
            failed_run.started_at = started_datetime
            failed_run.completed_at = datetime.now(timezone.utc)
            failed_run.duration_seconds = time.perf_counter() - started_at

        if failed_experiment is not None:
            failed_experiment.status = "failed"
            failed_experiment.updated_at = datetime.now(timezone.utc)

        try:
            session.commit()
        except Exception:
            session.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model training failed. Check the server logs.",
        ) from exc


# ---------------------------------------------------------
# 8. Predict using a completed model run
# ---------------------------------------------------------

@router.post("/runs/{run_id}/predict")
def predict_with_run(
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
        )
    ).first()

    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model run not found.",
        )

    if run.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Predictions are only available for completed runs.",
        )

    artifact = session.exec(
        select(Artifact).where(
            Artifact.run_id == run_id,
            Artifact.artifact_type == "model",
        )
    ).first()

    if artifact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trained model artifact not found.",
        )

    try:
        pipeline = load_model(
            file_path=artifact.file_path,
            expected_checksum=artifact.checksum_sha256,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Saved model file was not found.",
        ) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Saved model could not be loaded or verified.",
        ) from exc

    experiment = session.get(Experiment, run.experiment_id)

    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found.",
        )

    feature_columns = experiment.config.get("feature_columns", [])
    supplied_columns = set(request.input_data)
    required_columns = set(feature_columns)

    missing_columns = sorted(required_columns - supplied_columns)
    unexpected_columns = sorted(supplied_columns - required_columns)

    if missing_columns or unexpected_columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Input features do not match the model's expected features.",
                "missing_columns": missing_columns,
                "unexpected_columns": unexpected_columns,
            },
        )

    try:
        input_df = pd.DataFrame(
            [request.input_data],
            columns=feature_columns,
        )
        prediction_values = pipeline.predict(input_df)
        prediction = prediction_values[0]

        if hasattr(prediction, "item"):
            prediction = prediction.item()

        if hasattr(prediction, "isoformat"):
            prediction = prediction.isoformat()

    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Input values could not be processed by the trained model.",
        ) from exc

    prediction_record = Prediction(
        run_id=run_id,
        input_data=request.input_data,
        output_data={"prediction": prediction},
        status="success",
    )

    session.add(prediction_record)
    session.commit()
    session.refresh(prediction_record)

    return {
        "prediction_id": prediction_record.prediction_id,
        "run_id": run_id,
        "input_data": request.input_data,
        "prediction": prediction,
        "status": prediction_record.status,
        "created_at": prediction_record.created_at,
    }


# ---------------------------------------------------------
# 9. Prediction history for a run
# ---------------------------------------------------------

@router.get("/runs/{run_id}/predictions")
def get_prediction_history(
    run_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> dict:
    run = session.exec(
        select(ModelRun)
        .join(Experiment, ModelRun.experiment_id == Experiment.experiment_id)
        .where(
            ModelRun.run_id == run_id,
            Experiment.user_id == current_user.user_id,
        )
    ).first()

    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model run not found.",
        )

    predictions = session.exec(
        select(Prediction)
        .where(Prediction.run_id == run_id)
        .order_by(Prediction.created_at.desc())
    ).all()

    return {
        "run_id": run_id,
        "total": len(predictions),
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



@router.post("/{dataset_id}/compare", status_code=status.HTTP_201_CREATED)
def compare_dataset_models(
    dataset_id: int,
    request: ModelComparisonRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> dict:
    # 1. Validate dataset ownership and availability.
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

    file_path = Path(dataset.file_path)
    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Dataset file not found on the server.",
        )

    try:
        df = pd.read_csv(file_path)
    except (
        OSError,
        ValueError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the dataset CSV.",
        ) from exc

    if df.empty or df.columns.duplicated().any():
        raise HTTPException(
            status_code=400,
            detail="Dataset is empty or contains duplicate column names.",
        )

    if len(request.algorithms) != len(set(request.algorithms)):
        raise HTTPException(
            status_code=400,
            detail="Algorithms must be unique.",
        )

    # 2. Validate task/algorithm compatibility before creating an experiment.
    supported = ALGORITHMS.get(request.task_type, {})
    unsupported = [
        algorithm
        for algorithm in request.algorithms
        if algorithm not in supported
    ]
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Some algorithms do not support this task type.",
                "task_type": request.task_type,
                "unsupported_algorithms": unsupported,
                "available_algorithms": sorted(supported),
            },
        )

    required_columns = set(request.feature_columns)
    required_columns.add(request.target_column)
    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Requested columns do not exist.",
                "missing_columns": missing_columns,
            },
        )

    if (
        len(request.feature_columns) != len(set(request.feature_columns))
        or request.target_column in request.feature_columns
    ):
        raise HTTPException(
            status_code=400,
            detail="Features must be unique and must not include the target.",
        )

    # 3. Create one experiment to group all comparison runs.
    experiment = Experiment(
        user_id=current_user.user_id,
        dataset_id=dataset_id,
        name=f"{dataset.name} - Model Comparison",
        description="Comparison of multiple algorithms on one dataset.",
        task_type=request.task_type,
        target_column=request.target_column,
        status="running",
        config={
            "feature_columns": request.feature_columns,
            "test_size": request.test_size,
            "random_state": request.random_state,
            "algorithms": request.algorithms,
            "algorithm_hyperparameters": request.algorithm_hyperparameters,
        },
    )
    session.add(experiment)
    session.commit()
    session.refresh(experiment)

    results = []

    # 4. Train and persist each algorithm independently.
    for algorithm in request.algorithms:
        params = request.algorithm_hyperparameters.get(algorithm, {})
        started = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        run = ModelRun(
            experiment_id=experiment.experiment_id,
            algorithm=algorithm,
            status="running",
            hyperparameters={
                "test_size": request.test_size,
                "random_state": request.random_state,
                **params,
            },
            started_at=started_at,
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        try:
            training_result = train_model(
                df=df,
                feature_columns=request.feature_columns,
                target_column=request.target_column,
                task_type=request.task_type,
                algorithm=algorithm,
                test_size=request.test_size,
                random_state=request.random_state,
                hyperparameters=params,
            )

            model_info = save_model(
                pipeline=training_result["pipeline"],
                run_id=run.run_id,
            )

            run.status = "completed"
            run.model_version = model_info.get("model_version", "1")
            run.completed_at = datetime.now(timezone.utc)
            run.duration_seconds = time.perf_counter() - started

            session.add(
                Artifact(
                    experiment_id=experiment.experiment_id,
                    run_id=run.run_id,
                    name=model_info["name"],
                    artifact_type="model",
                    file_path=model_info["file_path"],
                    mime_type=model_info.get(
                        "mime_type", "application/octet-stream"
                    ),
                    file_size_bytes=model_info["file_size_bytes"],
                    checksum_sha256=model_info["checksum_sha256"],
                )
            )

            for metric_name, metric_value in training_result["metrics"].items():
                if metric_value is not None:
                    session.add(
                        ExperimentMetric(
                            run_id=run.run_id,
                            metric_name=metric_name,
                            metric_value=float(metric_value),
                            dataset_split="test",
                        )
                    )

            session.commit()
            session.refresh(run)

            artifact = session.exec(
                select(Artifact).where(Artifact.run_id == run.run_id)
            ).first()

            results.append({
                "run_id": run.run_id,
                "algorithm": algorithm,
                "status": "completed",
                "metrics": training_result["metrics"],
                "duration_seconds": run.duration_seconds,
                "artifact_id": artifact.artifact_id if artifact else None,
            })

        except Exception as exc:
            session.rollback()
            logger.exception(
                "Comparison training failed for algorithm=%s, dataset_id=%s",
                algorithm,
                dataset_id,
            )

            failed_run = session.get(ModelRun, run.run_id)
            if failed_run is not None:
                failed_run.status = "failed"
                failed_run.error_message = str(exc)[:1000]
                failed_run.completed_at = datetime.now(timezone.utc)
                failed_run.duration_seconds = time.perf_counter() - started
                session.add(failed_run)
                session.commit()

            results.append({
                "run_id": run.run_id,
                "algorithm": algorithm,
                "status": "failed",
                "metrics": {},
                "duration_seconds": time.perf_counter() - started,
                "error": str(exc)[:500],
            })

    # 5. Finalize the experiment and rank successful runs.
    successful = [item for item in results if item["status"] == "completed"]
    experiment.status = "completed" if successful else "failed"
    experiment.updated_at = datetime.now(timezone.utc)
    session.add(experiment)
    session.commit()

    ranking_metric = (
        "f1_weighted" if request.task_type == "classification" else "r2"
    )

    successful.sort(
        key=lambda item: (
            item["metrics"].get(ranking_metric) is not None,
            item["metrics"].get(ranking_metric) or 0,
        ),
        reverse=True,
    )

    for rank, item in enumerate(successful, start=1):
        item["rank"] = rank

    for item in results:
        if item["status"] == "failed":
            item["rank"] = None

    return {
        "message": "Model comparison finished.",
        "experiment_id": experiment.experiment_id,
        "dataset_id": dataset_id,
        "task_type": request.task_type,
        "ranking_metric": ranking_metric,
        "successful_runs": len(successful),
        "failed_runs": len(results) - len(successful),
        "results": sorted(
            results,
            key=lambda item: (
                item["rank"] is None,
                item["rank"] or 999,
            ),
        ),
    }
