import joblib
import mlflow

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RegressionTrainer:
    """
    End-to-end regression training, evaluation, selection, and persistence.
    """

    DEFAULT_CONFIG = {
        "linear": {},
        "ridge": {"alpha": 1.0},
        "lasso": {"alpha": 0.1},
        "rf": {"n_estimators": 10, "max_depth": 12, "n_jobs": -1},
        "gb": {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1},
    }

    def __init__(
        self,
        metric: str = "mae",
        config: dict | None = None,
        experiment_name: str = "NYC-Taxi-Trip-ML",
    ):
        self.metric = metric
        self.config = config or self.DEFAULT_CONFIG
        self.experiment_name = experiment_name

        self.best_model = None
        self.best_model_name = None
        self.best_score = self._initialize_score(metric)

        mlflow.set_experiment(self.experiment_name)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    @staticmethod
    def compute_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)

        try:
            rmse = mean_squared_error(y_true, y_pred, squared=False)
        except TypeError:
            rmse = mean_squared_error(y_true, y_pred) ** 0.5

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2_score(y_true, y_pred),
        }

    def evaluate(self, model, X, y):
        predictions = model.predict(X)
        return self.compute_metrics(y, predictions)

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------
    def build_models(self):
        return {
            "linear": LinearRegression(**self.config.get("linear", {})),
            "ridge": Ridge(**self.config.get("ridge", {})),
            "lasso": Lasso(**self.config.get("lasso", {})),
            "rf": RandomForestRegressor(**self.config.get("rf", {})),
            "gb": GradientBoostingRegressor(**self.config.get("gb", {})),
        }

    # ------------------------------------------------------------------
    # Model selection logic
    # ------------------------------------------------------------------
    @staticmethod
    def _is_better(metric, candidate, best):
        if metric in ("mae", "rmse"):
            return candidate < best
        return candidate > best

    @staticmethod
    def _initialize_score(metric):
        return float("inf") if metric in ("mae", "rmse") else -float("inf")

    # ------------------------------------------------------------------
    # Training pipeline
    # ------------------------------------------------------------------
    def train(self, X_train, y_train, X_valid, y_valid):
        models = self.build_models()

        for name, model in models.items():
            print(f"Training model: {name}")

            with mlflow.start_run(run_name=name):
                mlflow.log_params(self.config.get(name, {}))

                model.fit(X_train, y_train)
                metrics = self.evaluate(model, X_valid, y_valid)

                for key, value in metrics.items():
                    mlflow.log_metric(key, value)

                score = metrics[self.metric]
                if self._is_better(self.metric, score, self.best_score):
                    self.best_model = model
                    self.best_score = score
                    self.best_model_name = name

        print(
            f"Best model = {self.best_model_name} | "
            f"{self.metric.upper()} = {self.best_score:.4f}"
        )

        return self.best_model, self.best_model_name

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @staticmethod
    def save_model(model, filepath="best_model.pkl"):
        joblib.dump(model, filepath)
        return filepath

    @staticmethod
    def load_model(filepath="best_model.pkl"):
        return joblib.load(filepath)
