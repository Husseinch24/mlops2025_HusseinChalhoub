import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction import FeatureHasher


class FeatureEngineer:
    def __init__(self, max_ohe_cardinality=50, n_hash_bins=128):
        self.max_ohe_cardinality = max_ohe_cardinality
        self.n_hash_bins = n_hash_bins
        self._encoder = None
        self._scaler = None
        self._cat_columns_ = None
        self._num_columns_ = None

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
    def transform_categoricals(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = df.copy()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        if not cat_cols:
            return pd.DataFrame(index=df.index)

        df_cat = df[cat_cols].fillna("Unknown").astype(str)

        if fit or self._encoder is None:
            self._encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
            self._encoder.fit(df_cat)
            self._cat_columns_ = self._encoder.get_feature_names_out(cat_cols)

        encoded = self._encoder.transform(df_cat)

        return pd.DataFrame(encoded, columns=self._cat_columns_, index=df.index)


    # ---------------- Numeric ----------------
    def normalize_numeric(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = df.copy()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not num_cols:
            return pd.DataFrame(index=df.index)

        if fit or self._scaler is None:
            self._scaler = StandardScaler()
            self._scaler.fit(df[num_cols])
            self._num_columns_ = num_cols

        scaled = self._scaler.transform(df[self._num_columns_])
        return pd.DataFrame(scaled, columns=self._num_columns_, index=df.index)

    # ---------------- Full pipeline ----------------
    def feature_engineering(
    self,
    df: pd.DataFrame,
    fit: bool = False,
    save: bool = False,
    is_train: bool = True,
):

        df = self.add_distance_column(df)
        df = self.enrich_datetime(df)

        y = None
        if "trip_duration" in df.columns:
            y = df["trip_duration"].copy()
            df = df.drop(columns=["trip_duration"])

        num_df = self.normalize_numeric(df, fit=fit)
        cat_df = self.transform_categoricals(df, fit=fit)

        X = pd.concat([num_df, cat_df], axis=1)
        cols = X.columns.tolist()

        return X, y, cols
