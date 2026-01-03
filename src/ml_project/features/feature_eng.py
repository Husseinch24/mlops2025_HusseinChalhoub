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



    def transform_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Return a DataFrame with original categorical columns replaced by one-hot columns.

        Missing categorical values are filled with a sentinel so the encoder learns an explicit
        missing category. The method keeps all non-categorical columns intact.
        """
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

        # If no categorical columns, return dataframe unchanged
        if not cat_cols:
            return df

        # Fill missing values with a sentinel so the encoder creates a column for missing
        df_cat = df[cat_cols].fillna("__MISSING__")

        # Fit or transform encoding (use dense output to create a DataFrame)
        if fit or self._encoder is None:
            # sklearn changed the parameter name from `sparse` to `sparse_output` in newer versions
            try:
                self._encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                self._encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            encoded = self._encoder.fit_transform(df_cat)
        else:
            encoded = self._encoder.transform(df_cat)

        # Get feature names (compat with different sklearn versions)
        try:
            feat_names = self._encoder.get_feature_names_out(cat_cols)
        except AttributeError:
            feat_names = []
            for i, col in enumerate(cat_cols):
                cats = self._encoder.categories_[i]
                feat_names.extend([f"{col}_{c}" for c in cats])

        # Build DataFrame for encoded features and concatenate
        encoded_df = pd.DataFrame(encoded, columns=feat_names, index=df.index)

        df = df.drop(columns=cat_cols)
        df = pd.concat([df, encoded_df], axis=1)

        return df



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

        # ---- Apply categorical encoding first (returns DataFrame) ----
        df = self.transform_categoricals(df, fit=fit)

        # Extract target if present
        y = None
        if "trip_duration" in df.columns:
            y = pd.Series(df["trip_duration"].values, index=df.index)
            df = df.drop(columns=["trip_duration"], errors="ignore")

        X = df.copy()

        # ---- Normalize numeric columns in the DataFrame ----
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if fit or self._scaler is None:
            self._scaler = StandardScaler()
            if numeric_cols:
                X[numeric_cols] = self._scaler.fit_transform(X[numeric_cols])
        else:
            if numeric_cols:
                X[numeric_cols] = self._scaler.transform(X[numeric_cols])

        cols = X.columns.tolist()
        return X.reset_index(drop=True), y, cols

