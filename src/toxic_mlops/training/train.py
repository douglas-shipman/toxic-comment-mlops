import hashlib
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import wandb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from toxic_mlops.training.data import LABEL_COLUMNS, load_dataset, split_dataset

DATASET_PATH = Path("data/raw/train.csv")
MODEL_PATH = Path("models/toxic_comment_model.joblib")
MANIFEST_PATH = Path("data/processed/dataset_manifest.json")
DATASET_SHA256 = "bd4084611bd27c939ba98e5e63bc3e5a2c1a4e99477dcba46c829e4c986c429d"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
MAX_FEATURES = 50_000
THRESHOLD = 0.5


def calculate_sha256(path: Path) -> str:
    """Calculate the SHA-256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def get_git_commit() -> str:
    """Return the current Git commit or unknown when Git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def create_model() -> Pipeline:
    """Create the text vectorization and multilabel classification pipeline."""
    classifier = OneVsRestClassifier(
        LogisticRegression(
            class_weight="balanced",
            max_iter=300,
            random_state=RANDOM_STATE,
            solver="liblinear",
        ),
        n_jobs=1,
    )

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    max_features=MAX_FEATURES,
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def calculate_metrics(
    actual: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate aggregate and per-label validation metrics."""
    predicted = (probabilities >= THRESHOLD).astype(int)

    metrics = {
        "validation/f1_micro": f1_score(
            actual,
            predicted,
            average="micro",
            zero_division=0,
        ),
        "validation/f1_macro": f1_score(
            actual,
            predicted,
            average="macro",
            zero_division=0,
        ),
        "validation/precision_micro": precision_score(
            actual,
            predicted,
            average="micro",
            zero_division=0,
        ),
        "validation/recall_micro": recall_score(
            actual,
            predicted,
            average="micro",
            zero_division=0,
        ),
        "validation/roc_auc_macro": roc_auc_score(
            actual,
            probabilities,
            average="macro",
        ),
    }

    for index, label in enumerate(LABEL_COLUMNS):
        metrics[f"validation/{label}_f1"] = f1_score(
            actual[:, index],
            predicted[:, index],
            zero_division=0,
        )
        metrics[f"validation/{label}_roc_auc"] = roc_auc_score(
            actual[:, index],
            probabilities[:, index],
        )

    return metrics


def main() -> None:
    """Train, evaluate, save, and register the baseline model."""
    actual_hash = calculate_sha256(DATASET_PATH)

    if actual_hash != DATASET_SHA256:
        raise ValueError(
            f"Unexpected dataset hash. Expected {DATASET_SHA256}, got {actual_hash}"
        )

    dataframe = load_dataset(DATASET_PATH)
    train_data, validation_data = split_dataset(
        dataframe,
        validation_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    config = {
        "model": "tfidf_logistic_regression",
        "git_commit": get_git_commit(),
        "dataset_sha256": actual_hash,
        "dataset_rows": len(dataframe),
        "training_rows": len(train_data),
        "validation_rows": len(validation_data),
        "validation_size": VALIDATION_SIZE,
        "random_state": RANDOM_STATE,
        "max_features": MAX_FEATURES,
        "ngram_range": [1, 2],
        "class_weight": "balanced",
        "threshold": THRESHOLD,
        "labels": LABEL_COLUMNS,
    }

    with wandb.init(
        project="toxic-comment-mlops",
        job_type="training",
        config=config,
    ) as run:
        model = create_model()

        model.fit(
            train_data["comment_text"],
            train_data[LABEL_COLUMNS],
        )

        probabilities = model.predict_proba(validation_data["comment_text"])
        actual = validation_data[LABEL_COLUMNS].to_numpy()

        metrics = calculate_metrics(actual, probabilities)
        run.log(metrics)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "source": "Jigsaw Toxic Comment Classification Challenge",
            "file": str(DATASET_PATH),
            "sha256": actual_hash,
            "rows": len(dataframe),
            "columns": list(dataframe.columns),
        }
        MANIFEST_PATH.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        dataset_artifact = wandb.Artifact(
            name="jigsaw-toxic-comments",
            type="dataset",
            description="Manifest for the Jigsaw training dataset",
            metadata=manifest,
        )
        dataset_artifact.add_file(str(MANIFEST_PATH))
        run.log_artifact(dataset_artifact, aliases=["baseline"])

        model_artifact = wandb.Artifact(
            name="toxic-comment-classifier",
            type="model",
            description="TF-IDF and logistic-regression multilabel classifier",
            metadata=metrics,
        )
        model_artifact.add_file(str(MODEL_PATH))
        run.log_artifact(model_artifact, aliases=["candidate"])

        print("\nValidation metrics:")
        for name, value in sorted(metrics.items()):
            print(f"{name}: {value:.4f}")

        print(f"\nSaved model to: {MODEL_PATH}")
        print(f"W&B run: {run.url}")


if __name__ == "__main__":
    main()