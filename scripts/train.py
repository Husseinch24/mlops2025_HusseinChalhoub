import os
import joblib
import mlflow
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------------------------------------------
# Evaluation utilities
# -------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """Compute regression evaluation metrics.

    Support multiple scikit-learn versions: older versions do not accept the
    `squared` keyword in `mean_squared_error`, so gracefully fall back to
    computing RMSE from MSE when needed.
    """
    mae = mean_absolute_error(y_true, y_pred)

    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        # Older scikit-learn returns MSE and does not accept `squared` kwarg
        rmse = mean_squared_error(y_true, y_pred) ** 0.5

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred)
    }


def evaluate_model(model, X, y):
    """Run prediction and return evaluation metrics."""
    predictions = model.predict(X)
    return compute_metrics(y, predictions)


# -------------------------------------------------------------------
# Model factory
# -------------------------------------------------------------------
def build_models(config):
    """Instantiate all candidate models from configuration."""
    return {
        "linear": LinearRegression(**config.get("linear", {})),
        "ridge": Ridge(**config.get("ridge", {})),
        "lasso": Lasso(**config.get("lasso", {})),
        "rf": RandomForestRegressor(**config.get("rf", {})),
        "gb": GradientBoostingRegressor(**config.get("gb", {})),
    }


# -------------------------------------------------------------------
# Model selection logic
# -------------------------------------------------------------------
def is_better(metric, candidate, best):
    """Determine whether the candidate score is better."""
    if metric in ("mae", "rmse"):
        return candidate < best
    return candidate > best


def initialize_score(metric):
    """Initialize best score based on metric direction."""
    return float("inf") if metric in ("mae", "rmse") else -float("inf")


# -------------------------------------------------------------------
# Training pipeline
# -------------------------------------------------------------------
def train_models(X_train, y_train, X_valid, y_valid, metric="mae", config=None):
    default_config = {
        "linear": {},
        "ridge": {"alpha": 1.0},
        "lasso": {"alpha": 0.1},
        "rf": {"n_estimators": 10, "max_depth": 12, "n_jobs": -1},
        "gb": {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1},
    }

    cfg = config or default_config
    models = build_models(cfg)

    best_model = None
    best_name = None
    best_score = initialize_score(metric)

    mlflow.set_experiment("NYC-Taxi-Trip-ML")

    for name, model in models.items():
        print(f"Training model: {name}")

        with mlflow.start_run(run_name=name):
            mlflow.log_params(cfg.get(name, {}))

            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_valid, y_valid)

            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            score = metrics[metric]
            if is_better(metric, score, best_score):
                best_model = model
                best_score = score
                best_name = name

    print(f"Best model = {best_name} | {metric.upper()} = {best_score:.4f}")
    return best_model, best_name


# -------------------------------------------------------------------
# Persistence utilities
# -------------------------------------------------------------------
def save_model(model, filepath="best_model.pkl"):
    """Persist trained model to disk."""
    joblib.dump(model, filepath)
    return filepath


def load_model(filepath="best_model.pkl"):
    """Load a persisted model."""
    return joblib.load(filepath)
