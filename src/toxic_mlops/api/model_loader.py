import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

import wandb

LOCAL_MODEL_PATH = Path("models/toxic_comment_model.joblib")


@lru_cache
def get_model() -> tuple[Any, str]:
    """Load the production model from W&B or use the local model."""
    artifact_reference = os.getenv("WANDB_MODEL_ARTIFACT")

    if artifact_reference:
        artifact = wandb.Api().artifact(artifact_reference, type="model")
        download_directory = Path(
            artifact.download(root="models/wandb-production")
        )
        model_path = next(download_directory.rglob("*.joblib"))
        return joblib.load(model_path), artifact_reference

    if not LOCAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Local model not found and WANDB_MODEL_ARTIFACT is not configured."
        )

    return joblib.load(LOCAL_MODEL_PATH), "local-development-model"