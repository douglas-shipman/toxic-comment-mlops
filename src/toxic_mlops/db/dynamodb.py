import os
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import boto3

DEFAULT_TABLE_NAME = "toxic-comment-predictions"
DEFAULT_REGION = "us-east-1"


def is_dynamodb_enabled() -> bool:
    """Return whether persistent prediction logging is enabled."""
    return os.getenv("DYNAMODB_ENABLED", "false").lower() == "true"


def get_table():
    """Return the configured DynamoDB table."""
    table_name = os.getenv("DYNAMODB_TABLE_NAME", DEFAULT_TABLE_NAME)
    region = os.getenv("AWS_REGION", DEFAULT_REGION)

    dynamodb = boto3.resource("dynamodb", region_name=region)
    return dynamodb.Table(table_name)


def log_prediction(
    request_id: str,
    comment_text: str,
    is_toxic: bool,
    labels: list[dict[str, Any]],
    model_version: str,
    latency_ms: float,
) -> None:
    """Persist one prediction request and its result."""
    label_data = {
        item["label"]: {
            "probability": Decimal(str(item["probability"])),
            "predicted": item["predicted"],
        }
        for item in labels
    }

    predicted_classes = [
        item["label"]
        for item in labels
        if item["predicted"]
    ]

    get_table().put_item(
        Item={
            "request_id": request_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "comment_text": comment_text,
            "is_toxic": is_toxic,
            "labels": label_data,
            "predicted_classes": predicted_classes,
            "model_version": model_version,
            "latency_ms": Decimal(str(latency_ms)),
            "feedback_received": False,
        }
    )

def record_feedback(
    request_id: str,
    actual_labels: list[str],
) -> bool:
    """Save human feedback and return whether the prediction was correct."""
    table = get_table()
    response = table.get_item(Key={"request_id": request_id})
    item = response.get("Item")

    if item is None:
        raise KeyError(f"Prediction not found: {request_id}")

    predicted_classes = set(item.get("predicted_classes", []))
    correct_classes = set(actual_labels)
    prediction_correct = predicted_classes == correct_classes

    table.update_item(
        Key={"request_id": request_id},
        UpdateExpression=(
            "SET feedback_received = :received, "
            "feedback_timestamp = :timestamp, "
            "actual_labels = :labels, "
            "prediction_correct = :correct"
        ),
        ExpressionAttributeValues={
            ":received": True,
            ":timestamp": datetime.now(UTC).isoformat(),
            ":labels": sorted(correct_classes),
            ":correct": prediction_correct,
        },
    )

    return prediction_correct