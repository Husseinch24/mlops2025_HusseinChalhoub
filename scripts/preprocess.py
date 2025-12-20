import pandas as pd
import numpy as np

# --- Validation ---
def validate_columns(df, required_cols, is_train=True):
    """Ensure all required columns exist."""
    if is_train:
        required_cols = required_cols + ['trip_duration']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

# --- Missing values ---
def handle_missing_values(df):
    """Drop missing location data and fill missing passenger_count."""
    location_cols = ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']
    # Make an explicit copy after dropping to avoid chained-assignment issues
    df = df.dropna(subset=location_cols).copy()
    if 'passenger_count' in df.columns:
        # Avoid inplace fill on a possible view; assign back instead
        df['passenger_count'] = df['passenger_count'].fillna(1)
    return df

# --- Coordinate validation ---
def filter_invalid_coordinates(df):
    """Remove rows with coordinates outside valid ranges."""
    valid = (
        df['pickup_latitude'].between(-90, 90) &
        df['dropoff_latitude'].between(-90, 90) &
        df['pickup_longitude'].between(-180, 180) &
        df['dropoff_longitude'].between(-180, 180)
    )
    return df[valid]

# --- Trip duration cleaning ---
def clean_trip_duration(df):
    """Remove negative or zero durations and outliers."""
    df = df[df['trip_duration'] > 0]
    lower, upper = df['trip_duration'].quantile([0.01, 0.99])
    return df[df['trip_duration'].between(lower, upper)]

# --- DateTime processing ---
def process_datetime(df):
    """Convert strings to datetime and drop invalid rows."""
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    if 'dropoff_datetime' in df.columns:
        df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], errors='coerce')
    df = df.dropna(subset=['pickup_datetime'])
    return df

def add_time_features(df):
    """Extract hour, day, weekday, month from pickup_datetime."""
    dt = df['pickup_datetime']
    df['pickup_hour'] = dt.dt.hour
    df['pickup_day'] = dt.dt.day
    df['pickup_weekday'] = dt.dt.weekday
    df['pickup_month'] = dt.dt.month
    return df

# --- Duplicates ---
def remove_dupes(df):
    """Remove duplicate rows, preferring 'id' if present."""
    subset = ['id'] if 'id' in df.columns else None
    return df.drop_duplicates(keep='first', subset='id')

# --- Full preprocessing pipeline ---
def preprocess(df, is_train=True):
    required_cols = ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','pickup_datetime']
    df = validate_columns(df, required_cols, is_train=is_train)
    df = handle_missing_values(df)
    df = filter_invalid_coordinates(df)
    
    if is_train:
        df = clean_trip_duration(df)
    
    df = remove_dupes(df)
    df = process_datetime(df)
    df = add_time_features(df)
    return df.reset_index(drop=True)
