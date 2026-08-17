import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

LABEL_NAMES = {
    "toxic": "Toxic",
    "severe_toxic": "Severely toxic",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity hate",
}

st.set_page_config(
    page_title="Toxic Comment Moderator",
    page_icon="🛡️",
    layout="centered",
)

st.title("🛡️ Toxic Comment Moderator")
st.write(
    "Analyze a comment for six forms of toxicity and provide feedback "
    "to help monitor the model."
)

if "prediction" not in st.session_state:
    st.session_state.prediction = None

comment = st.text_area(
    "Comment",
    height=160,
    placeholder="Enter a comment to analyze...",
)

if st.button("Analyze comment", type="primary", use_container_width=True):
    if not comment.strip():
        st.warning("Enter a comment before submitting.")
    else:
        try:
            response = httpx.post(
                f"{API_URL}/predict",
                json={"comment_text": comment},
                timeout=30,
            )
            response.raise_for_status()
            st.session_state.prediction = response.json()
        except httpx.HTTPError as error:
            st.error(f"The prediction service is unavailable: {error}")

prediction = st.session_state.prediction

if prediction:
    if prediction["is_toxic"]:
        st.error("Potentially toxic content detected")
    else:
        st.success("No toxicity labels crossed the decision threshold")

    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric(
        "Overall result",
        "Toxic" if prediction["is_toxic"] else "Non-toxic",
    )
    metric_two.metric(
        "Latency",
        f'{prediction["latency_ms"]:.1f} ms',
    )
    metric_three.metric(
        "Labels detected",
        sum(item["predicted"] for item in prediction["labels"]),
    )

    probabilities = pd.DataFrame(
        [
            {
                "Label": LABEL_NAMES[item["label"]],
                "Probability": item["probability"],
            }
            for item in prediction["labels"]
        ]
    )

    st.subheader("Label probabilities")
    st.bar_chart(
        probabilities.set_index("Label"),
        horizontal=True,
    )

    detected = [
        LABEL_NAMES[item["label"]]
        for item in prediction["labels"]
        if item["predicted"]
    ]

    st.write(
        "**Detected labels:** "
        + (", ".join(detected) if detected else "None")
    )
    st.caption(
        f'Model: {prediction["model_version"]} · '
        f'Request: {prediction["request_id"]}'
    )

    st.divider()
    st.subheader("Human review")
    st.write(
        "Select the labels that should apply. Leave all labels "
        "unselected for a non-toxic comment."
    )

    actual_labels = st.multiselect(
        "Correct labels",
        options=list(LABEL_NAMES),
        format_func=lambda label: LABEL_NAMES[label],
    )

    if st.button("Submit feedback", use_container_width=True):
        try:
            response = httpx.post(
                f"{API_URL}/feedback",
                json={
                    "request_id": prediction["request_id"],
                    "actual_labels": actual_labels,
                },
                timeout=30,
            )
            response.raise_for_status()
            feedback = response.json()

            if feedback["prediction_correct"]:
                st.success("Feedback saved. The prediction was correct.")
            else:
                st.warning("Feedback saved. The prediction was corrected.")
        except httpx.HTTPError as error:
            st.error(f"Unable to save feedback: {error}")