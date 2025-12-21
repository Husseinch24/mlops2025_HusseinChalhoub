import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# scikit-learn is an optional dependency in some environments; skip these tests if it's missing
pytest.importorskip("sklearn")

from scripts.batch_inference import batch_inference


class DummyModel:
    def __init__(self, preds):
        self.preds = np.array(preds)

    def predict(self, X):
        # return values truncated/padded to match X length
        n = len(X)
        return self.preds[:n]


def _fake_feature_engineering(df, fit=False, save=False, is_train=False):
    # return a simple numeric DataFrame X that matches the input length
    return pd.DataFrame({"f": np.arange(len(df))}, index=df.index), None, ["f"]


def test_batch_inference_adds_prediction_and_timestamp(monkeypatch):
    monkeypatch.setattr("scripts.batch_inference.feature_engineering", _fake_feature_engineering)

    model = DummyModel([10.0, 20.0, 30.0])
    df = pd.DataFrame({"id": [1, 2, 3]})

    out = batch_inference(model, df.copy(), save_path=None)

    assert "prediction" in out.columns
    assert "timestamp" in out.columns

    # predictions preserved and correspond to model output
    assert out["prediction"].tolist() == [10.0, 20.0, 30.0]

    # timestamp column should be datetime-like
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"]) or isinstance(out["timestamp"].iloc[0], datetime)


def test_batch_inference_saves_to_given_filepath(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.batch_inference.feature_engineering", _fake_feature_engineering)

    model = DummyModel([1.0, 2.0])
    df = pd.DataFrame({"id": [1, 2]})

    out_file = tmp_path / "my_preds.csv"
    out = batch_inference(model, df.copy(), save_path=str(out_file))

    assert out_file.exists()

    saved = pd.read_csv(out_file)
    assert "prediction" in saved.columns
    assert saved["prediction"].tolist() == [1.0, 2.0]
    assert "timestamp" in saved.columns


def test_batch_inference_saves_to_directory_with_date_prefix(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.batch_inference.feature_engineering", _fake_feature_engineering)

    model = DummyModel([5.0, 6.0])
    df = pd.DataFrame({"id": [1, 2]})

    out_dir = tmp_path
    out = batch_inference(model, df.copy(), save_path=str(out_dir))

    # expect a single file named <YYYYMMDD>_predictions.csv
    files = list(out_dir.glob("*_predictions.csv"))
    assert len(files) == 1

    fname = files[0].name
    date_prefix = datetime.now().strftime("%Y%m%d")
    assert fname.startswith(date_prefix)

    saved = pd.read_csv(files[0])
    assert "prediction" in saved.columns
    assert saved["prediction"].tolist() == [5.0, 6.0]
    assert "timestamp" in saved.columns


if __name__ == "__main__":
    pytest.main([__file__])
