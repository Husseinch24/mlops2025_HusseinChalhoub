import pandas as pd
import numpy as np
import pytest

from scripts.preprocess import (
    validate_columns,
    handle_missing_values,
    filter_invalid_coordinates,
    clean_trip_duration,
    process_datetime,
    add_time_features,
    remove_dupes,
    preprocess,
)


def test_validate_columns_raises_for_missing_train_col():
    df = pd.DataFrame(columns=['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','pickup_datetime'])
    with pytest.raises(ValueError):
        validate_columns(df, ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','pickup_datetime'], is_train=True)


def test_validate_columns_no_raise_for_non_train():
    df = pd.DataFrame(columns=['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','pickup_datetime'])
    out = validate_columns(df, ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','pickup_datetime'], is_train=False)
    assert out is df


def test_handle_missing_values_drops_and_fills():
    df = pd.DataFrame({
        'pickup_latitude':[10.0, 20.0, np.nan],
        'pickup_longitude':[10.0, 20.0, 30.0],
        'dropoff_latitude':[11.0, 21.0, 31.0],
        'dropoff_longitude':[12.0, 22.0, 32.0],
        'passenger_count':[np.nan, 2, 3]
    })
    out = handle_missing_values(df)
    assert len(out) == 2  # third row with missing pickup coords dropped
    assert out['passenger_count'].iloc[0] == 1  # NaN filled with 1


def test_filter_invalid_coordinates_removes_bad_rows():
    df = pd.DataFrame({
        'pickup_latitude':[10, 100],
        'pickup_longitude':[10, 10],
        'dropoff_latitude':[20, 20],
        'dropoff_longitude':[30, 30],
    })
    out = filter_invalid_coordinates(df)
    assert len(out) == 1
    assert out['pickup_latitude'].iloc[0] == 10


def test_clean_trip_duration_removes_non_positive_and_outliers():
    durations = list(range(1, 201)) + [10000, 0, -1]
    df = pd.DataFrame({'trip_duration': durations})
    out = clean_trip_duration(df)
    assert (out['trip_duration'] > 0).all()
    assert 10000 not in out['trip_duration'].values
    assert 0 not in out['trip_duration'].values
    assert -1 not in out['trip_duration'].values


def test_process_datetime_parses_and_drops_invalid():
    df = pd.DataFrame({'pickup_datetime': ['2020-01-01 00:00:00', 'not a date']})
    out = process_datetime(df)
    assert len(out) == 1
    assert pd.api.types.is_datetime64_any_dtype(out['pickup_datetime'])


def test_add_time_features_creates_expected_columns():
    df = pd.DataFrame({'pickup_datetime': pd.to_datetime(['2020-01-02 03:04:05'])})
    out = add_time_features(df.copy())
    assert out.at[0, 'pickup_hour'] == 3
    assert out.at[0, 'pickup_day'] == 2
    assert out.at[0, 'pickup_weekday'] == 3  # 2020-01-02 is a Thursday -> weekday 3
    assert out.at[0, 'pickup_month'] == 1


def test_remove_dupes_prefers_id():
    df = pd.DataFrame({'id':[1,1,2], 'val':[10,10,20]})
    out = remove_dupes(df)
    assert len(out) == 2
    assert set(out['id']) == {1,2}


def test_preprocess_integration():
    df = pd.DataFrame({
        'id': [1, 1, 2],
        'pickup_latitude':[10,10,20],
        'pickup_longitude':[10,10,20],
        'dropoff_latitude':[11,11,21],
        'dropoff_longitude':[11,11,21],
        'pickup_datetime':['2020-01-01 01:00:00','2020-01-01 01:00:00','2020-01-02 02:00:00'],
        'trip_duration':[100,100,100],
        'passenger_count':[1,1,2]
    })
    out = preprocess(df, is_train=True)
    # Duplicate id 1 should be removed -> 2 rows
    assert len(out) == 2
    for col in ['pickup_hour','pickup_day','pickup_weekday','pickup_month']:
        assert col in out.columns


if __name__ == '__main__':
    pytest.main([__file__])
