import os
import sys
# ensure the src package is importable when running tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
import numpy as np
import pytest

from ml_project.preprocess.preprocessor import Preprocessor


def test_validate_schema_raises_for_missing_train_col():
    pre = Preprocessor(is_train=True)
    df = pd.DataFrame(columns=['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','pickup_datetime'])
    with pytest.raises(ValueError):
        pre.validate_schema(df)


def test_validate_schema_no_raise_for_non_train():
    pre = Preprocessor(is_train=False)
    df = pd.DataFrame(columns=['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','pickup_datetime'])
    out = pre.validate_schema(df)
    assert out is df


def test_handle_missing_drops_and_fills():
    pre = Preprocessor()
    df = pd.DataFrame({
        'pickup_latitude':[10.0, 20.0, np.nan],
        'pickup_longitude':[10.0, 20.0, 30.0],
        'dropoff_latitude':[11.0, 21.0, 31.0],
        'dropoff_longitude':[12.0, 22.0, 32.0],
        'passenger_count':[np.nan, 2, 3]
    })
    out = pre.handle_missing(df)
    assert len(out) == 2  # third row with missing pickup coords dropped
    assert out['passenger_count'].iloc[0] == 1  # NaN filled with 1


def test_filter_coordinates_removes_bad_rows():
    pre = Preprocessor()
    df = pd.DataFrame({
        'pickup_latitude':[10, 100],
        'pickup_longitude':[10, 10],
        'dropoff_latitude':[20, 20],
        'dropoff_longitude':[30, 30],
    })
    out = pre.filter_coordinates(df)
    assert len(out) == 1
    assert out['pickup_latitude'].iloc[0] == 10


def test_clean_duration_removes_non_positive_and_outliers():
    pre = Preprocessor(is_train=True)
    durations = list(range(1, 201)) + [10000, 0, -1]
    df = pd.DataFrame({'trip_duration': durations})
    out = pre.clean_duration(df)
    assert (out['trip_duration'] > 0).all()
    assert 10000 not in out['trip_duration'].values
    assert 0 not in out['trip_duration'].values
    assert -1 not in out['trip_duration'].values


def test_process_datetime_parses_and_creates_features():
    pre = Preprocessor()
    df = pd.DataFrame({'pickup_datetime': ['2020-01-01 00:00:00', 'not a date']})
    out = pre.process_datetime(df)
    assert len(out) == 1
    assert pd.api.types.is_datetime64_any_dtype(out['pickup_datetime'])
    # feature columns
    for col in ['pickup_hour', 'pickup_day', 'pickup_weekday', 'pickup_month']:
        assert col in out.columns


def test_deduplicate_prefers_id_if_present():
    pre = Preprocessor()
    df = pd.DataFrame({'id':[1,1,2], 'val':[10,10,20]})
    out = pre.deduplicate(df)
    assert len(out) == 2
    assert set(out['id']) == {1,2}


def test_deduplicate_without_id_removes_exact_duplicates():
    pre = Preprocessor()
    df = pd.DataFrame({'a':[1,1], 'b':[2,2]})
    out = pre.deduplicate(df)
    assert len(out) == 1


def test_run_integration_train_pipeline():
    pre = Preprocessor(is_train=True)
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
    out = pre.run(df)
    # Duplicate id 1 should be removed -> 2 rows
    assert len(out) == 2
    for col in ['pickup_hour','pickup_day','pickup_weekday','pickup_month']:
        assert col in out.columns


def test_run_non_train_does_not_require_trip_duration_and_skips_duration_filter():
    pre = Preprocessor(is_train=False)
    df = pd.DataFrame({
        'id': [1, 1, 2],
        'pickup_latitude':[10,10,20],
        'pickup_longitude':[10,10,20],
        'dropoff_latitude':[11,11,21],
        'dropoff_longitude':[11,11,21],
        'pickup_datetime':['2020-01-01 01:00:00','2020-01-01 01:00:00','2020-01-02 02:00:00'],
        'passenger_count':[1,1,2]
    })
    # Should not raise for missing trip_duration and should process
    out = pre.run(df)
    assert len(out) == 2


if __name__ == '__main__':
    pytest.main([__file__])
