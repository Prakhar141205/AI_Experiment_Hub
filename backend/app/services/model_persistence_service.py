
import hashlib
import pickle
from pathlib import Path
from typing import Any
from uuid import uuid4


MODEL_STORAGE_DIR = Path("uploads/models")


def save_model(
    pipeline: Any,
    run_id: int,
) -> dict[str, Any]:
    """Save a fitted ML pipeline and return its file metadata."""

    MODEL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"run_{run_id}_{uuid4().hex}.pkl"
    file_path = MODEL_STORAGE_DIR / filename

    # Serialize the fitted pipeline.
    with file_path.open("wb") as file:
        pickle.dump(pipeline, file, protocol=pickle.HIGHEST_PROTOCOL)

    # Calculate file size and SHA-256 checksum.
    file_size_bytes = file_path.stat().st_size
    checksum = hashlib.sha256(file_path.read_bytes()).hexdigest()

    return {
        "name": filename,
        "artifact_type": "trained_model",
        "file_path": str(file_path),
        "mime_type": "application/octet-stream",
        "file_size_bytes": file_size_bytes,
        "checksum_sha256": checksum,
    }


def load_model(
    file_path: str,
    expected_checksum: str,
) -> Any:
    """Load a trusted saved pipeline after verifying its checksum."""

    model_path = Path(file_path)

    if not model_path.is_file():
        raise FileNotFoundError("Saved model artifact was not found.")

    actual_checksum = hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()

    if actual_checksum != expected_checksum:
        raise ValueError("Model artifact checksum verification failed.")

    with model_path.open("rb") as file:
        return pickle.load(file)
