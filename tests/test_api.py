import numpy as np
from fastapi.testclient import TestClient

from toxic_mlops.api import main as api_main


class FakeModel:
    def predict_proba(self, comments):
        assert len(comments) == 1
        return np.array([[0.9, 0.1, 0.2, 0.05, 0.7, 0.1]])


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "is_model_configured",
        lambda: True,
    )
    client = TestClient(api_main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_available": True,
    }


def test_predict_endpoint(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "get_model",
        lambda: (FakeModel(), "test-model:v1"),
    )
    client = TestClient(api_main.app)

    response = client.post(
        "/predict",
        json={"comment_text": "Example comment"},
    )

    assert response.status_code == 200

    result = response.json()
    assert result["model_version"] == "test-model:v1"
    assert result["is_toxic"] is True
    assert result["labels"][0]["label"] == "toxic"
    assert result["labels"][0]["predicted"] is True
    assert result["labels"][4]["label"] == "insult"
    assert result["labels"][4]["predicted"] is True
    assert result["latency_ms"] >= 0


def test_predict_rejects_empty_comment():
    client = TestClient(api_main.app)

    response = client.post(
        "/predict",
        json={"comment_text": ""},
    )

    assert response.status_code == 422

def test_feedback_endpoint(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "is_dynamodb_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        api_main,
        "record_feedback",
        lambda request_id, actual_labels: True,
    )
    client = TestClient(api_main.app)

    response = client.post(
        "/feedback",
        json={
            "request_id": "test-request",
            "actual_labels": ["toxic", "insult"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "request_id": "test-request",
        "saved": True,
        "prediction_correct": True,
    }


def test_feedback_rejects_unknown_labels(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "is_dynamodb_enabled",
        lambda: True,
    )
    client = TestClient(api_main.app)

    response = client.post(
        "/feedback",
        json={
            "request_id": "test-request",
            "actual_labels": ["not_a_real_label"],
        },
    )

    assert response.status_code == 422