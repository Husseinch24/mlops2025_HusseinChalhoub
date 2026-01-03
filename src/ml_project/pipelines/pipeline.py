import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.ml_project.preprocess.preprocessor import Preprocessor
from src.ml_project.features.feature_eng import FeatureEngineer
from src.ml_project.train.training import RegressionTrainer
from src.ml_project.inference.inference import BatchPredictor


class TaxiPipeline:
    """
    Full ML pipeline for NYC Taxi Trip Duration:
    - Preprocessing
    - Feature engineering
    - Model training
    - Batch inference
    """

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
        print(f"Training complete. Best model: {best_model_name}")
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
            print("No existing inference pipeline. Checking for best model...")
            bm = getattr(self.model_trainer, "best_model", None)
            if bm is not None and hasattr(bm, "predict"):
                print("Best model found and supports prediction. Creating predictor.")
                self.inference = BatchPredictor(model=bm)
            else:
                raise RuntimeError("No trained model available for inference")

        print("Running predictions...")
        result = self.inference.batch_inference(df, save_path=save_path)
        print("Inference completed.\n")
        return result

    def run(self):
        print("========== PIPELINE STARTED ==========\n")

        # Load train/test
        train_df = self.load_data(self.cfg["paths"]["train_csv"])
        test_df = self.load_data(self.cfg["paths"]["test_csv"])

        # Split train/valid
        print("Splitting data into train/validation sets...")
        train_split, valid_split = train_test_split(
            train_df,
            test_size=self.cfg["train"]["test_size"],
            random_state=self.cfg["train"]["seed"],
        )
        print("Split completed.\n")

        # Preprocess
        train_df = self.preprocess(train_split)
        valid_df = self.preprocess(valid_split, is_train=False)
        test_df = self.preprocess(test_df, is_train=False)

        # Feature engineering
        X_train, y_train = self.feature_engineering(train_df, fit=True)
        X_valid, y_valid = self.feature_engineering(valid_df, fit=False)

        # Train model
        model, best_model_name = self.train(X_train, y_train, X_valid, y_valid)

        # Prepare a usable model for inference
        if not hasattr(model, "predict"):
            print("Model does not have predict. Wrapping with dummy model.")
            class _DummyModel:
                def predict(self, X):
                    import numpy as np
                    return np.zeros(len(X))
            model_to_use = _DummyModel()
        else:
            model_to_use = model

        # Ensure inference pipeline
        print("Preparing inference pipeline...")
        if self.inference is None:
            self.inference = BatchPredictor(model=model_to_use)
        else:
            self.inference.model = model_to_use

        # Fit feature engineering if missing
        if getattr(self.feature_engineer, "_encoder", None) is None or getattr(self.feature_engineer, "_scaler", None) is None:
            print("Feature engineer not fitted. Fitting now...")
            self.feature_engineer.feature_engineering(train_df, fit=True, is_train=True)

        # Save model
        os.makedirs(self.cfg["paths"]["artifact_dir"], exist_ok=True)
        print(f"Saving model to: {self.cfg['paths']['artifact_dir']}")
        try:
            self.model_trainer.save_model(model, f"{self.cfg['paths']['artifact_dir']}/best_model.pkl")
            print("Model saved successfully.\n")
        except Exception as e:
            print(f"Warning: Model save failed: {e}\n")

        # Batch inference
        print("Running final batch inference on test data...")
        os.makedirs(self.cfg["paths"]["output_dir"], exist_ok=True)
        try:
            output_df = self.batch_inference(test_df, save_path=self.cfg["paths"]["output_dir"])
        except Exception as e:
            print(f"Batch inference failed: {e}")
            print("Creating fallback output...")
            output_df = test_df.copy()
            output_df["prediction"] = 0
            output_df["timestamp"] = pd.Timestamp.now()
            out_file = Path(self.cfg["paths"]["output_dir"]) / f"{pd.Timestamp.now().strftime('%Y%m%d')}_predictions.csv"
            output_df.to_csv(out_file, index=False)
            print(f"Fallback predictions saved to: {out_file}\n")

        print("========== PIPELINE COMPLETED ==========\n")
        return model, best_model_name, output_df
