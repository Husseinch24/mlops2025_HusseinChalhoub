import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler




class FeatureEngineer:
    def __init__(self):
        self._encoder = None
        self._scaler = None

    # ------------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------------
    @staticmethod
    def compute_haversine(lat1, lon1, lat2, lon2):
        """
        Compute great-circle distance between points (km).
        """
        lat1, lon1, lat2, lon2 = map(
            np.radians, [lat1, lon1, lat2, lon2]
        )

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        earth_radius_km = 6371.0
        return earth_radius_km * c

    def add_distance_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df["distance_km"] = self.compute_haversine(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )

        # Fill missing values with median
        median = df["distance_km"].median()
        df["distance_km"] = df["distance_km"].fillna(median)

        return df

    # ------------------------------------------------------------------
    # Datetime enrichment
    # ------------------------------------------------------------------
    def enrich_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        df["pickup_datetime"] = pd.to_datetime(
            df["pickup_datetime"], errors="coerce"
        )

        df["pickup_hour"] = df["pickup_datetime"].dt.hour
        df["pickup_day"] = df["pickup_datetime"].dt.day
        df["pickup_weekday"] = df["pickup_datetime"].dt.weekday
        df["pickup_month"] = df["pickup_datetime"].dt.month

        # Fill invalid dates with median values
        for col in [
            "pickup_hour",
            "pickup_day",
            "pickup_weekday",
            "pickup_month",
        ]:
            median = df[col].median()
            df[col] = df[col].fillna(median)

        return df

    # ------------------------------------------------------------------
    # Categorical encoding
    # ------------------------------------------------------------------



    def transform_categoricals(self, df, fit):
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

        # If no categoricals, simply return X,y
        if not cat_cols:
            y = df["trip_duration"].values if "trip_duration" in df.columns else None
            df = df.drop(columns=["trip_duration"], errors="ignore")
            X = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32").values
            return X, y

        # Fit or transform encoding
        if fit or self._encoder is None:
            self._encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            encoded = self._encoder.fit_transform(df[cat_cols])
        else:
            encoded = self._encoder.transform(df[cat_cols])

        # Drop original categorical columns
        df = df.drop(columns=cat_cols)

        # Extract target if present
        y = df["trip_duration"].values if "trip_duration" in df.columns else None
        df = df.drop(columns=["trip_duration"], errors="ignore")

        # Convert numeric columns to valid dtype
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
        numeric_data = df.values  # OK now

        # Combine dense numeric + sparse categorical
        X = sparse.hstack([numeric_data, encoded], format="csr")

        return X, y



    # ------------------------------------------------------------------
    # Numeric normalization
    # ------------------------------------------------------------------
    def normalize_numeric(
        self, df: pd.DataFrame, fit: bool = False
    ) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if fit or self._scaler is None:
            self._scaler = StandardScaler()
            df[numeric_cols] = self._scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self._scaler.transform(df[numeric_cols])

        return df

    # ------------------------------------------------------------------
    # End-to-end pipeline
    # ------------------------------------------------------------------
    def feature_engineering(
        self,
        df: pd.DataFrame,
        fit: bool = False,
        save: bool = False,
        is_train: bool = True,
    ):
        df = self.add_distance_column(df)
        df = self.enrich_datetime(df)
        # ---- Apply categorical encoding first ----
        X, y = self.transform_categoricals(df, fit=fit)

        # ---- Normalize numeric columns inside the sparse matrix ----
        # standard scaler must be applied before hstack, so we do this on raw numeric data
        if fit or self._scaler is None:
            self._scaler = StandardScaler(with_mean=False)  # sparse-safe
            X = self._scaler.fit_transform(X)
        else:
            X = self._scaler.transform(X)

        # There are no DataFrame columns anymore, so return feature names from encoder + numeric
        cols = []
        return X, y, cols

