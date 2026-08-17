from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from toxic_mlops.api.model_loader import LOCAL_MODEL_PATH, get_model
from toxic_mlops.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    LabelPrediction,
    PredictionRequest,
    PredictionResponse,
)
from toxic_mlops.db.dynamodb import (
    is_dynamodb_enabled,
    log_prediction,
    record_feedback,
)
from toxic_mlops.training.data import LABEL_COLUMNS

PREDICTION_THRESHOLD = 0.5

app = FastAPI(
    title="Toxic Comment Classification API",
    description="Multilabel toxicity predictions using a registered ML model.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report whether the API and local model are available."""
    return HealthResponse(
        status="healthy",
        model_available=LOCAL_MODEL_PATH.exists(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Classify a comment across the six toxicity labels."""
    started_at = perf_counter()

    try:
        model, model_version = get_model()
        probabilities = model.predict_proba([request.comment_text])[0]
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is unavailable.",
        ) from error

    labels = [
        LabelPrediction(
            label=label,
            probability=round(float(probability), 6),
            predicted=bool(probability >= PREDICTION_THRESHOLD),
        )
        for label, probability in zip(
            LABEL_COLUMNS,
            probabilities,
            strict=True,
        )
    ]

    latency_ms = (perf_counter() - started_at) * 1000

    response = PredictionResponse(
        request_id=str(uuid4()),
        is_toxic=any(item.predicted for item in labels),
        labels=labels,
        model_version=model_version,
        latency_ms=round(latency_ms, 3),
    )

    if is_dynamodb_enabled():
        try:
            log_prediction(
                request_id=response.request_id,
                comment_text=request.comment_text,
                is_toxic=response.is_toxic,
                labels=[item.model_dump() for item in response.labels],
                model_version=response.model_version,
                latency_ms=response.latency_ms,
            )
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail="Prediction logging is unavailable.",
            ) from error

    return response

@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Attach human-reviewed labels to an existing prediction."""
    if not is_dynamodb_enabled():
        raise HTTPException(
            status_code=503,
            detail="Feedback storage is unavailable.",
        )

    invalid_labels = sorted(set(request.actual_labels) - set(LABEL_COLUMNS))
    if invalid_labels:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown labels: {invalid_labels}",
        )

    try:
        prediction_correct = record_feedback(
            request_id=request.request_id,
            actual_labels=request.actual_labels,
        )
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail="Prediction request not found.",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Feedback storage is unavailable.",
        ) from error

    return FeedbackResponse(
        request_id=request.request_id,
        saved=True,
        prediction_correct=prediction_correct,
    )