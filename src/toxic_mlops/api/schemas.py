from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    comment_text: str = Field(min_length=1, max_length=5000)


class LabelPrediction(BaseModel):
    label: str
    probability: float
    predicted: bool


class PredictionResponse(BaseModel):
    request_id: str
    is_toxic: bool
    labels: list[LabelPrediction]
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_available: bool

class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1)
    actual_labels: list[str] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    request_id: str
    saved: bool
    prediction_correct: bool