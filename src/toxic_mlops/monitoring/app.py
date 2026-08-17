import os

import boto3
import pandas as pd
import streamlit as st

from toxic_mlops.training.data import LABEL_COLUMNS

TABLE_NAME = os.getenv(
    "DYNAMODB_TABLE_NAME",
    "toxic-comment-predictions",
)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

st.set_page_config(
    page_title="Model Monitoring",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(ttl=30)
def load_predictions(
    table_name: str,
    region: str,
) -> pd.DataFrame:
    """Read all available prediction logs from DynamoDB."""
    table = boto3.resource(
        "dynamodb",
        region_name=region,
    ).Table(table_name)

    items = []
    response = table.scan()
    items.extend(response.get("Items", []))

    while "LastEvaluatedKey" in response:
        response = table.scan(
            ExclusiveStartKey=response["LastEvaluatedKey"]
        )
        items.extend(response.get("Items", []))

    rows = [
        {
            "request_id": item["request_id"],
            "timestamp": item["timestamp"],
            "latency_ms": float(item["latency_ms"]),
            "is_toxic": bool(item["is_toxic"]),
            "predicted_classes": item.get("predicted_classes", []),
            "feedback_received": bool(
                item.get("feedback_received", False)
            ),
            "prediction_correct": item.get("prediction_correct"),
            "model_version": item.get("model_version", "unknown"),
        }
        for item in items
    ]

    dataframe = pd.DataFrame(rows)

    if not dataframe.empty:
        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"],
            utc=True,
        )
        dataframe = dataframe.sort_values("timestamp")

    return dataframe


st.title("📊 Toxic Comment Model Monitoring")
st.write(
    "Production telemetry read directly from the DynamoDB prediction log."
)

if st.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()

try:
    predictions = load_predictions(TABLE_NAME, AWS_REGION)
except Exception as error:
    st.error(f"Unable to read monitoring data: {error}")
    st.stop()

if predictions.empty:
    st.info("No prediction records are available yet.")
    st.stop()

feedback = predictions[predictions["feedback_received"]]
live_accuracy = (
    feedback["prediction_correct"].astype(bool).mean()
    if not feedback.empty
    else None
)

metric_one, metric_two, metric_three, metric_four = st.columns(4)

metric_one.metric("Predictions", len(predictions))
metric_two.metric(
    "Average latency",
    f'{predictions["latency_ms"].mean():.1f} ms',
)
metric_three.metric("Feedback records", len(feedback))
metric_four.metric(
    "Live accuracy",
    f"{live_accuracy:.1%}" if live_accuracy is not None else "No data",
)

st.subheader("Prediction latency over time")
latency_chart = predictions.set_index("timestamp")[["latency_ms"]]
st.line_chart(latency_chart)

st.subheader("Predicted class distribution")
exploded_labels = predictions["predicted_classes"].explode()
class_counts = (
    exploded_labels.dropna()
    .value_counts()
    .reindex(LABEL_COLUMNS, fill_value=0)
    .rename_axis("Label")
    .to_frame("Predictions")
)
st.bar_chart(class_counts)

st.subheader("Target distribution")
target_distribution = pd.DataFrame(
    {
        "Classification": ["Non-toxic", "Toxic"],
        "Predictions": [
            int((~predictions["is_toxic"]).sum()),
            int(predictions["is_toxic"].sum()),
        ],
    }
).set_index("Classification")
st.bar_chart(target_distribution)

st.subheader("Recent prediction activity")
recent = predictions[
    [
        "timestamp",
        "request_id",
        "is_toxic",
        "predicted_classes",
        "latency_ms",
        "feedback_received",
        "prediction_correct",
        "model_version",
    ]
].tail(20)

st.dataframe(
    recent,
    use_container_width=True,
    hide_index=True,
)