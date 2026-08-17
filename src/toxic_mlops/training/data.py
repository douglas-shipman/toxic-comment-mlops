from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

REQUIRED_COLUMNS = ["id", "comment_text", *LABEL_COLUMNS]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the Jigsaw training dataset."""
    data_path = Path(path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    dataframe = pd.read_csv(data_path)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    dataframe = dataframe[REQUIRED_COLUMNS].copy()
    dataframe["comment_text"] = dataframe["comment_text"].fillna("").astype(str)
    dataframe[LABEL_COLUMNS] = dataframe[LABEL_COLUMNS].astype(int)

    return dataframe


def split_dataset(
    dataframe: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a reproducible split stratified by any toxic label."""
    has_any_toxic_label = dataframe[LABEL_COLUMNS].max(axis=1)

    train_data, validation_data = train_test_split(
        dataframe,
        test_size=validation_size,
        random_state=random_state,
        stratify=has_any_toxic_label,
    )

    return train_data.reset_index(drop=True), validation_data.reset_index(drop=True)