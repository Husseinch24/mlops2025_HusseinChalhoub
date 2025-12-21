import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_encoder = None
_scaler = None

# -------------------------------------------------------------------
# Distance computation
# -------------------------------------------------------------------
def compute_haversine(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance in kilometers."""
    earth_radius = 6371.0088
    coords = map(np.radians, [lat1, lon1, lat2, lon2])
    lat1, lon1, lat2, lon2 = coords

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    h = (
        np.sin(delta_lat / 2) ** 2 +
        np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )

    return 2 * earth_radius * np.arctan2(np.sqrt(h), np.sqrt(1 - h))


def add_distance_column(df, batch=100_000):
    """Compute pickup–dropoff distance in chunks."""
    dist = np.empty(len(df))

    for i in range(0, len(df), batch):
        j = i + batch
        dist[i:j] = compute_haversine(
            df.loc[i:j, 'pickup_latitude'],
            df.loc[i:j, 'pickup_longitude'],
            df.loc[i:j, 'dropoff_latitude'],
            df.loc[i:j, 'dropoff_longitude'],
        )

    df['distance_km'] = pd.Series(dist, index=df.index)
    df['distance_km'].fillna(df['distance_km'].median(), inplace=True)
    return df


# -------------------------------------------------------------------
# Datetime features
# -------------------------------------------------------------------
def enrich_datetime(df):
    """Extract temporal features from pickup_datetime."""
    dt = pd.to_datetime(df['pickup_datetime'], errors='coerce')

    features = {
        'pickup_hour': dt.dt.hour,
        'pickup_day': dt.dt.day,
        'pickup_weekday': dt.dt.weekday,
        'pickup_month': dt.dt.month,
    }

    for name, values in features.items():
        df[name] = values.fillna(values.median())

    df['pickup_datetime'] = dt
    return df


# -------------------------------------------------------------------
# Categorical encoding
# -------------------------------------------------------------------
def transform_categoricals(df, fit=False):
    """One-hot encode categorical columns."""
    global _encoder

    categories = ['vendor_id', 'store_and_fwd_flag']
    df[categories] = df[categories].fillna("Unknown")

    if fit or _encoder is None:
        _encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        matrix = _encoder.fit_transform(df[categories])
    else:
        matrix = _encoder.transform(df[categories])

    encoded = pd.DataFrame(
        matrix,
        index=df.index,
        columns=_encoder.get_feature_names_out(categories)
    )

    return pd.concat([df.drop(columns=categories), encoded], axis=1)


# -------------------------------------------------------------------
# Numeric scaling
# -------------------------------------------------------------------
def normalize_numeric(df, fit=False):
    """Standardize numeric feature columns."""
    global _scaler

    numeric = [
        'distance_km',
        'pickup_hour',
        'pickup_day',
        'pickup_weekday',
        'pickup_month',
        'passenger_count'
    ]

    if fit or _scaler is None:
        _scaler = StandardScaler()
        df[numeric] = _scaler.fit_transform(df[numeric])
    else:
        df[numeric] = _scaler.transform(df[numeric])

    return df


# -------------------------------------------------------------------
# Full feature pipeline
# -------------------------------------------------------------------
def feature_engineering(df, fit=False, save=False, is_train=True):
    """End-to-end feature engineering pipeline."""
    df = (
        df
        .pipe(add_distance_column)
        .pipe(enrich_datetime)
        .pipe(transform_categoricals, fit=fit)
        .pipe(normalize_numeric, fit=fit)
    )

    drop_cols = ['trip_duration', 'id', 'pickup_datetime', 'dropoff_datetime']
    X = df.drop(columns=drop_cols, errors='ignore')

    y = df['trip_duration'] if is_train and 'trip_duration' in df else None

    if save:
        df.to_csv("processed_features.csv", index=False)

    return X, y, X.columns.tolist()
