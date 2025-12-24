
import os
import sys
import numpy as np
import pytest

# ensure the src package is importable when running tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sklearn.linear_model import LinearRegression

import mlflow

from ml_project.train.training import RegressionTrainer


def test_compute_metrics_basic():
	y_true = np.array([1.0, 2.0, 3.0])
	y_pred = np.array([1.0, 2.0, 4.0])

	metrics = RegressionTrainer.compute_metrics(y_true, y_pred)

	assert set(metrics.keys()) == {"mae", "rmse", "r2"}
	assert pytest.approx(metrics["mae"]) == (abs(1 - 1) + abs(2 - 2) + abs(3 - 4)) / 3


def test_initialize_score_and_is_better():
	# mae/rmse should initialize to +inf and smaller is better
	assert RegressionTrainer._initialize_score("mae") == float("inf")
	assert RegressionTrainer._is_better("mae", 1.0, 2.0)
	assert not RegressionTrainer._is_better("mae", 3.0, 2.0)

	# other metrics initialize to -inf and larger is better
	assert RegressionTrainer._initialize_score("r2") == -float("inf")
	assert RegressionTrainer._is_better("r2", 0.9, 0.5)


def test_build_models_respects_config():
	# avoid mlflow side effects by monkeypatching set_experiment
	mlflow.set_experiment = lambda *a, **k: None

	cfg = {"rf": {"n_estimators": 3, "max_depth": 2, "n_jobs": 1}}
	trainer = RegressionTrainer(config=cfg)
	models = trainer.build_models()

	assert "rf" in models
	rf = models["rf"]
	assert getattr(rf, "n_estimators", None) == 3


def test_train_selects_best_model(monkeypatch):
	# monkeypatch mlflow functions to avoid external side effects
	monkeypatch.setattr(mlflow, "set_experiment", lambda *a, **k: None)

	class DummyRun:
		def __enter__(self):
			return self

		def __exit__(self, exc_type, exc, tb):
			return False

	monkeypatch.setattr(mlflow, "start_run", lambda *a, **k: DummyRun())
	monkeypatch.setattr(mlflow, "log_params", lambda *a, **k: None)
	monkeypatch.setattr(mlflow, "log_metric", lambda *a, **k: None)

	# tiny synthetic dataset (linear)
	X = np.arange(20).reshape(-1, 1).astype(float)
	y = (2 * X.squeeze()) + 1

	X_train, X_valid = X[:12], X[12:16]
	y_train, y_valid = y[:12], y[12:16]

	cfg = {
		"linear": {},
		"ridge": {"alpha": 0.1},
		"lasso": {"alpha": 0.1},
		"rf": {"n_estimators": 2, "max_depth": 2, "n_jobs": 1},
		"gb": {"n_estimators": 2, "max_depth": 2, "learning_rate": 0.1},
	}

	trainer = RegressionTrainer(metric="mae", config=cfg)
	best_model, best_name = trainer.train(X_train, y_train, X_valid, y_valid)

	assert best_model is not None
	assert best_name in {"linear", "ridge", "lasso", "rf", "gb"}


def test_save_and_load_model(tmp_path):
	model = LinearRegression()
	path = tmp_path / "temp_model.pkl"

	fp = RegressionTrainer.save_model(model, filepath=str(path))
	assert fp == str(path)

	loaded = RegressionTrainer.load_model(filepath=str(path))
	assert hasattr(loaded, "predict")


if __name__ == "__main__":
	pytest.main([__file__])

