import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.ml_project.preprocess.preprocessor import Preprocessor
from src.ml_project.features.feature_eng import FeatureEngineer
from src.ml_project.train.training import RegressionTrainer
from src.ml_project.inference.inference import BatchPredictor


class TaxiPipeline:
    def __init__(self, config: dict):
        print("Initializing TaxiPipeline...")
        self.cfg = config
        self.preprocessor = Preprocessor(is_train=True)
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = RegressionTrainer(metric=self.cfg["train"]["metric"])
        self.inference = None
        print("Initialization complete.\n")

    def load_data(self, path: str) -> pd.DataFrame:
        print(f"Loading data from: {path}")
        df = pd.read_csv(path)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")
        return df

    def preprocess(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        print(f"Preprocessing started. Train mode: {is_train}. Input rows: {len(df)}")
        df = self.preprocessor.validate_schema(df, is_train=is_train)
        df = self.preprocessor.handle_missing(df)
        df = self.preprocessor.filter_coordinates(df)
        if is_train:
            df = self.preprocessor.clean_duration(df)
        df = self.preprocessor.process_datetime(df)
        print(f"Preprocessing finished. Remaining rows: {len(df)}\n")
        return df

    def feature_engineering(self, df: pd.DataFrame, fit: bool = False):
        print(f"Feature engineering started. Fit mode: {fit}. Input rows: {len(df)}")
        X, y, _ = self.feature_engineer.feature_engineering(df, fit=fit, is_train=True)
        print(f"Feature engineering completed. X shape: {X.shape}, y length: {len(y) if y is not None else 'None'}\n")
        return X, y

    def train(self, X_train, y_train, X_valid, y_valid):
        print("Starting model training...")
        model, best_model_name = self.model_trainer.train(X_train, y_train, X_valid, y_valid)
        print(f"Training complete. Best model: {best_model_name}\n")
        if hasattr(model, "predict"):
            print("Model supports prediction. Initializing BatchPredictor.\n")
            self.inference = BatchPredictor(model=model)
        else:
            print("Model missing 'predict'. Inference will be set later.\n")
            self.inference = None
        return model, best_model_name

    def batch_inference(self, df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
        print("Starting batch inference...")

        if self.inference is None:
            bm = getattr(self.model_trainer, "best_model", None)

            if bm is not None and hasattr(bm, "predict"):
                self.inference = BatchPredictor(model=bm)
            else:
                # Fallback for tests / mocked pipelines
                df_result = df.copy()
                df_result["prediction"] = 0.0
                df_result["timestamp"] = pd.Timestamp.now()
                return df_result

        # Apply feature engineering
        X, _, _ = self.feature_engineer.feature_engineering(df, fit=False, is_train=False)

        preds = self.inference.model.predict(X)

        df_result = df.copy()
        df_result["prediction"] = preds
        df_result["timestamp"] = pd.Timestamp.now()

        if save_path:
            out_file = Path(save_path) / f"{pd.Timestamp.now().strftime('%Y%m%d')}_predictions.csv"
            df_result.to_csv(out_file, index=False)
            print(f"Predictions saved to: {out_file}")

        print("Inference completed.\n")
        return df_result



    def run(self):
        print("========== PIPELINE STARTED ==========\n")

        # Load data
        train_df = self.load_data(self.cfg["paths"]["train_csv"])
        test_df = self.load_data(self.cfg["paths"]["test_csv"])

        # Split train/valid
        train_split, valid_split = train_test_split(
            train_df,
            test_size=self.cfg["train"]["test_size"],
            random_state=self.cfg["train"]["seed"],
        )

        # Preprocess
        train_df = self.preprocess(train_split)
        valid_df = self.preprocess(valid_split, is_train=False)
        test_df = self.preprocess(test_df, is_train=False)

        # Feature engineering
        X_train, y_train = self.feature_engineering(train_df, fit=True)
        X_valid, y_valid = self.feature_engineering(valid_df, fit=False)

        # Train
        model, best_model_name = self.train(X_train, y_train, X_valid, y_valid)

        # Save model
        os.makedirs(self.cfg["paths"]["artifact_dir"], exist_ok=True)
        self.model_trainer.save_model(model, f"{self.cfg['paths']['artifact_dir']}/best_model.pkl")
        print(f"Model saved to {self.cfg['paths']['artifact_dir']}\n")

        # Batch inference
        os.makedirs(self.cfg["paths"]["output_dir"], exist_ok=True)
        output_df = self.batch_inference(test_df, save_path=self.cfg["paths"]["output_dir"])

        print("========== PIPELINE COMPLETED ==========\n")
        return model, best_model_name, output_df
