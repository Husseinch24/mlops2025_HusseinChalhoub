import pandas as pd
import numpy as np


class Preprocessor:
    """
    Preprocessing pipeline for the NYC Taxi dataset.
    Performs:
    - schema validation
    - missing value handling
    - coordinate sanity checks
    - datetime parsing and feature extraction
    - duration filtering (train only)
    - duplicate removal
    """

    def __init__(self, is_train: bool = True):
        self.is_train = is_train

    # ----------------------------------------------------
    # Validation
    # ----------------------------------------------------
    def _required_columns(self, is_train: bool = True) -> set:
        base = {
            'pickup_latitude',
            'pickup_longitude',
            'dropoff_latitude',
            'dropoff_longitude',
            'pickup_datetime'
        }
        base.add('trip_duration') if is_train else None
        return base

    def validate_schema(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        missing = self._required_columns(is_train).difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        return df

    # ----------------------------------------------------
    # Missing values
    # ----------------------------------------------------
    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(
            subset=[
                'pickup_latitude',
                'pickup_longitude',
                'dropoff_latitude',
                'dropoff_longitude'
            ]
        )

        if 'passenger_count' in df:
            df = df.assign(
                passenger_count=df['passenger_count'].fillna(1)
            )

        return df

    # ----------------------------------------------------
    # Coordinate cleaning
    # ----------------------------------------------------
    def filter_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        bounds = {
            'pickup_latitude': (-90, 90),
            'dropoff_latitude': (-90, 90),
            'pickup_longitude': (-180, 180),
            'dropoff_longitude': (-180, 180),
        }

        mask = np.logical_and.reduce(
            [df[col].between(low, high) for col, (low, high) in bounds.items()]
        )

        return df.loc[mask]

    # ----------------------------------------------------
    # Trip duration
    # ----------------------------------------------------
    def clean_duration(self, df: pd.DataFrame,
                       lower_q: float = 0.01,
                       upper_q: float = 0.99) -> pd.DataFrame:
        df = df[df['trip_duration'] > 0]

        lo, hi = df['trip_duration'].quantile([lower_q, upper_q])
        return df[df['trip_duration'].between(lo, hi)]

    # ----------------------------------------------------
    # Datetime features
    # ----------------------------------------------------
    def process_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['pickup_datetime'] = pd.to_datetime(
            df['pickup_datetime'], errors='coerce'
        )

        if 'dropoff_datetime' in df:
            df['dropoff_datetime'] = pd.to_datetime(
                df['dropoff_datetime'], errors='coerce'
            )

        df = df.dropna(subset=['pickup_datetime'])

        dt = df['pickup_datetime']
        df = df.assign(
            pickup_hour=dt.dt.hour,
            pickup_day=dt.dt.day,
            pickup_weekday=dt.dt.weekday,
            pickup_month=dt.dt.month
        )

        return df

    # ----------------------------------------------------
    # Duplicates
    # ----------------------------------------------------
    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        key = ['id'] if 'id' in df else None
        return df.drop_duplicates(subset=key)

    # ----------------------------------------------------
    # Orchestrator
    # ----------------------------------------------------
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = (
            df
            .pipe(self.validate_schema)
            .pipe(self.handle_missing)
            .pipe(self.filter_coordinates)
        )

        if self.is_train:
            df = df.pipe(self.clean_duration)

        df = (
            df
            .pipe(self.deduplicate)
            .pipe(self.process_datetime)
            .reset_index(drop=True)
        )

        return df
    