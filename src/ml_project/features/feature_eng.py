import numpy as np
import pandas as pd

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
    def transform_categoricals(
        self, df: pd.DataFrame, fit: bool = False
    ) -> pd.DataFrame:
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

        if not cat_cols:
            return df

        if fit or self._encoder is None:
            self._encoder = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            )
            encoded = self._encoder.fit_transform(df[cat_cols])
        else:
            encoded = self._encoder.transform(df[cat_cols])

        encoded_df = pd.DataFrame(
            encoded,
            columns=self._encoder.get_feature_names_out(cat_cols),
            index=df.index,
        )

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
        df = self.transform_categoricals(df, fit=fit)

        y = None
        if is_train and "trip_duration" in df.columns:
            # extract target before normalizing so y remains in original units
            y = df["trip_duration"].copy()
            df = df.drop(columns=["trip_duration"])

        df = self.normalize_numeric(df, fit=fit)

        cols = df.columns.tolist()

        return df, y, cols
