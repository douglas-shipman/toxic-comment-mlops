import pandas as pd
import pytest

from toxic_mlops.training.data import LABEL_COLUMNS, load_dataset, split_dataset


def make_sample_data() -> pd.DataFrame:
    rows = []

    for index in range(20):
        is_toxic = int(index % 2 == 0)
        rows.append(
            {
                "id": str(index),
                "comment_text": f"Sample comment {index}",
                "toxic": is_toxic,
                "severe_toxic": 0,
                "obscene": 0,
                "threat": 0,
                "insult": 0,
                "identity_hate": 0,
            }
        )

    return pd.DataFrame(rows)


def test_load_dataset(tmp_path):
    path = tmp_path / "train.csv"
    make_sample_data().to_csv(path, index=False)

    dataframe = load_dataset(path)

    assert len(dataframe) == 20
    assert list(dataframe.columns) == ["id", "comment_text", *LABEL_COLUMNS]


def test_load_dataset_rejects_missing_columns(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"comment_text": ["hello"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        load_dataset(path)


def test_split_dataset_is_reproducible():
    dataframe = make_sample_data()

    train_one, validation_one = split_dataset(dataframe)
    train_two, validation_two = split_dataset(dataframe)

    assert len(train_one) == 16
    assert len(validation_one) == 4
    assert train_one["id"].tolist() == train_two["id"].tolist()
    assert validation_one["id"].tolist() == validation_two["id"].tolist()
    assert set(train_one["id"]).isdisjoint(validation_one["id"])