import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher


class FeatureEngineer:
    """
    Feature engineering for NYC Taxi Trip Duration.
    - Distance calculation
    - Datetime enrichment
    - Categorical encoding (OneHot or Hashing)
    - Numeric scaling
    """

    def __init__(self, max_ohe_cardinality=50, n_hash_bins=128):
        self._encoder = None
        self._scaler = None
        self.max_ohe_cardinality = max_ohe_cardinality
        self.n_hash_bins = n_hash_bins

    # ---------------- Distance ----------------
    @staticmethod
    def compute_haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371.0 * c

    def add_distance_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["distance_km"] = self.compute_haversine(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )
        df["distance_km"] = df["distance_km"].fillna(df["distance_km"].median())
        return df

    # ---------------- Datetime ----------------
    def enrich_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        df["pickup_hour"] = df["pickup_datetime"].dt.hour.fillna(0).astype(int)
        df["pickup_day"] = df["pickup_datetime"].dt.day.fillna(1).astype(int)
        df["pickup_weekday"] = df["pickup_datetime"].dt.weekday.fillna(0).astype(int)
        df["pickup_month"] = df["pickup_datetime"].dt.month.fillna(1).astype(int)
        return df

    # ---------------- Categorical ----------------
    def transform_categoricals(self, df: pd.DataFrame, fit: bool = False) -> sparse.csr_matrix | None:
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if not cat_cols:
            return None

        df_cat = df[cat_cols].fillna("__MISSING__").astype(str)

        # Low cardinality -> OneHotEncoder
        low_card_cols = [c for c in cat_cols if df[c].nunique() <= self.max_ohe_cardinality]
        high_card_cols = [c for c in cat_cols if df[c].nunique() > self.max_ohe_cardinality]

        features = []

        if low_card_cols:
            df_low = df_cat[low_card_cols]
            if fit or self._encoder is None:
                self._encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
                features.append(self._encoder.fit_transform(df_low))
            else:
                features.append(self._encoder.transform(df_low))

        if high_card_cols:
            df_high = df_cat[high_card_cols]
            hasher = FeatureHasher(n_features=self.n_hash_bins, input_type='string')
            hashed = hasher.transform(df_high.astype(str).values)
            features.append(hashed)

        if features:
            return sparse.hstack(features, format="csr")
        return None

    # ---------------- Numeric ----------------
    def normalize_numeric(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray | None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return None
        if fit or self._scaler is None:
            self._scaler = StandardScaler()
            return self._scaler.fit_transform(df[numeric_cols])
        return self._scaler.transform(df[numeric_cols])

    # ---------------- Full pipeline ----------------
    def feature_engineering(self, df: pd.DataFrame, fit: bool = False, is_train: bool = True):
        print(f"[FeatureEngineer] Starting feature engineering. Fit={fit}, Rows={len(df)}")
        df = self.add_distance_column(df)
        df = self.enrich_datetime(df)

        y = None
        if "trip_duration" in df.columns:
            y = df["trip_duration"].copy()
            df = df.drop(columns=["trip_duration"], errors="ignore")

        numeric_scaled = self.normalize_numeric(df, fit=fit)
        cat_encoded = self.transform_categoricals(df, fit=fit)

        features = []
        if numeric_scaled is not None:
            features.append(sparse.csr_matrix(numeric_scaled))
        if cat_encoded is not None:
            features.append(cat_encoded)

        X_sparse = sparse.hstack(features, format="csr") if features else None
        print(f"[FeatureEngineer] Completed feature engineering. X shape={X_sparse.shape if X_sparse is not None else None}, y length={len(y) if y is not None else 'None'}")
        return X_sparse, y, df
