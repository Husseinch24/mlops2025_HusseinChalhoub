import os
import pandas as pd
from datetime import datetime
from pathlib import Path
import joblib
from sklearn.base import BaseEstimator  # for type hinting if you want

class BatchPredictor:
    def __init__(self, model: BaseEstimator = None):
        """
        Initialize the BatchPredictor with a model.
        :param model: A trained scikit-learn-like model
        """
        self.model = model

    @staticmethod
    def feature_engineering(df: pd.DataFrame, fit: bool = False, save: bool = False, is_train: bool = False):
        """Simple, stable feature engineering used for inference tests.

        Keeps all non-target columns as features and returns X (DataFrame), y (Series or None),
        and feature column names.
        """
        feature_cols = [col for col in df.columns if col != 'target']
        X = df[feature_cols].copy()
        y = df['target'] if 'target' in df.columns else None
        return X, y, feature_cols

    def batch_inference(self, df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
        """
        Perform batch inference on a dataframe.
        :param df: Input dataframe
        :param save_path: Directory or file path to save predictions
        :return: DataFrame with predictions and timestamp
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please load a model before inference.")
        
        X, _, feature_cols = self.feature_engineering(df, fit=False, save=False, is_train=False)
        preds = self.model.predict(X)
        
        df['prediction'] = preds
        df['timestamp'] = datetime.now()

        if save_path:
            if os.path.isdir(save_path):
                date_str = datetime.now().strftime("%Y%m%d")
                save_file = os.path.join(save_path, f"{date_str}_predictions.csv")
            else:
                save_file = save_path
            df.to_csv(save_file, index=False)
            print(f"Predictions saved to {save_file}")

        return df

    @staticmethod
    def load_model(filepath: str = "best_model.pkl"):
        """
        Load a persisted model from file.
        :param filepath: Path to saved model
        :return: Loaded model
        """
        return joblib.load(filepath)
