import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import sparse
import numpy as np
from typing import Tuple


class RegressionTrainer:
    DEFAULT_CONFIG = {
        "linear": {},
        "ridge": {"alpha": 1.0},
        "rf": {"n_estimators": 50, "max_depth": 12, "n_jobs": -1},
        "gb": {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1},
    }

    def __init__(self, metric="mae", config=None):
        self.metric = metric
        self.config = config or self.DEFAULT_CONFIG
        self.best_model = None
        self.best_model_name = None
        self.best_score = float("inf") if metric in ("mae", "rmse") else -float("inf")
        self.model_metrics = {}

    @staticmethod
    def compute_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {"mae": mae, "rmse": rmse, "r2": r2}

    def evaluate(self, model, X, y):
        if sparse.issparse(X):
            X_eval = X.toarray()
        else:
            X_eval = X
        return self.compute_metrics(y, model.predict(X_eval))

    def build_models(self):
        return {
            "linear": LinearRegression(**self.config.get("linear", {})),
            "ridge": Ridge(**self.config.get("ridge", {})),
            "rf": RandomForestRegressor(**self.config.get("rf", {})),
            "gb": GradientBoostingRegressor(**self.config.get("gb", {})),
        }

    def _is_better(self, metric, candidate, best):
        if metric in ("mae", "rmse"):
            return candidate < best
        return candidate > best

    def train(self, X_train, y_train, X_valid, y_valid) -> Tuple[object, str]:
        models = self.build_models()

        # Only convert to dense for tree-based models if features < 2000
        n_features = X_train.shape[1]
        X_train_dense = X_train.toarray() if sparse.issparse(X_train) and n_features <= 2000 else X_train
        X_valid_dense = X_valid.toarray() if sparse.issparse(X_valid) and n_features <= 2000 else X_valid

        for name, model in models.items():
            print(f"[Trainer] Training model: {name}")
            if name in ["rf", "gb"] and sparse.issparse(X_train):
                if n_features > 2000:
                    print(f"⚠️ Skipping {name}: too many features for dense tree models ({n_features})")
                    continue
                X_tr, X_val = X_train_dense, X_valid_dense
            else:
                X_tr, X_val = X_train, X_valid

            try:
                model.fit(X_tr, y_train)
                metrics = self.evaluate(model, X_val, y_valid)
                self.model_metrics[name] = metrics
                if self._is_better(self.metric, metrics[self.metric], self.best_score):
                    self.best_model = model
                    self.best_model_name = name
                    self.best_score = metrics[self.metric]
            except Exception as e:
                print(f"⚠️ Training failed for {name}: {e}")

        print(f"[Trainer] Best model = {self.best_model_name} | {self.metric.upper()} = {self.best_score:.4f}")
        print("[Trainer] All model metrics:")
        for k, v in self.model_metrics.items():
            print(f" - {k}: {v}")
        return self.best_model, self.best_model_name

    @staticmethod
    def save_model(model, filepath="best_model.pkl"):
        joblib.dump(model, filepath)
        return filepath

    @staticmethod
    def load_model(filepath="best_model.pkl"):
        return joblib.load(filepath)
