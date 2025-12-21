import os
import numpy as np
import pandas as pd
import pytest
import contextlib

# joblib is required for persistence tests; skip this module if it's not available in the environment
pytest.importorskip("joblib")

import scripts.train as train


def test_compute_metrics_basic():
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.0, 1.5, 1.8])

    metrics = train.compute_metrics(y_true, y_pred)

    assert set(metrics.keys()) == {"mae", "rmse", "r2"}
    assert pytest.approx(metrics["mae"], rel=1e-6) == np.mean(np.abs(y_true - y_pred))


def test_evaluate_model_uses_model_predict():
    class DummyModel:
        def predict(self, X):
            return np.zeros(len(X))

    model = DummyModel()
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = np.array([0.0, 0.0, 0.0])

    metrics = train.evaluate_model(model, X, y)
    assert metrics["mae"] == 0.0


def test_build_models_returns_expected_keys_and_instances():
    cfg = {"ridge": {"alpha": 0.5}, "rf": {"n_estimators": 2}}
    models = train.build_models(cfg)

    assert set(models.keys()) == {"linear", "ridge", "lasso", "rf", "gb"}
    # Check at least one concrete type
    from sklearn.linear_model import Ridge
    assert isinstance(models["ridge"], Ridge)


@pytest.mark.parametrize(
    "metric,candidate,best,expected",
    [
        ("mae", 0.1, 0.2, True),
        ("rmse", 0.1, 0.2, True),
        ("r2", 0.9, 0.8, True),
        ("r2", 0.1, 0.8, False),
    ],
)
def test_is_better(metric, candidate, best, expected):
    assert train.is_better(metric, candidate, best) is expected


@pytest.mark.parametrize("metric,init", [("mae", float("inf")), ("rmse", float("inf")), ("r2", -float("inf"))])
def test_initialize_score(metric, init):
    assert train.initialize_score(metric) == init


def test_train_models_selects_best_with_mocked_models(monkeypatch):
    # Mock mlflow to avoid side effects
    monkeypatch.setattr(train, "mlflow", type("M", (), {
        "set_experiment": lambda *a, **k: None,
        # start_run should be usable as a context manager in the production code
        "start_run": lambda *a, **k: contextlib.nullcontext(),
        "log_params": lambda *a, **k: None,
        "log_metric": lambda *a, **k: None,
    }))

    # Create fake models with deterministic predictions
    class FakeModel:
        def __init__(self, preds):
            self.preds = np.array(preds)

        def fit(self, X, y):
            # no-op
            return self

        def predict(self, X):
            return self.preds[: len(X)]

    def fake_build_models(cfg):
        # 'best' model will have predictions closest to y_valid below
        return {
            "a": FakeModel([10.0, 10.0]),
            "b": FakeModel([1.0, 1.0]),
            "c": FakeModel([5.0, 5.0]),
        }

    monkeypatch.setattr(train, "build_models", fake_build_models)

    X_train = pd.DataFrame({"x": [1, 2]})
    y_train = np.array([0.0, 0.0])
    X_valid = pd.DataFrame({"x": [1, 2]})
    y_valid = np.array([1.0, 1.0])

    best_model, best_name = train.train_models(X_train, y_train, X_valid, y_valid, metric="mae", config={})

    assert best_name == "b"
    assert isinstance(best_model, FakeModel)


def test_save_and_load_model_roundtrip(tmp_path):
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    # Train a trivial model so it can be persisted
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    model.fit(X, y)

    dest = tmp_path / "mymodel.pkl"
    path = train.save_model(model, filepath=str(dest))

    assert os.path.exists(path)

    loaded = train.load_model(filepath=path)
    assert isinstance(loaded, LinearRegression)
    # Predictions should be approximately equal
    preds = loaded.predict(np.array([[4.0]]))
    assert pytest.approx(preds[0], rel=1e-6) == 8.0
