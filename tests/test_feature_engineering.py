import pandas as pd
import numpy as np
import pytest

# sklearn is an optional dependency in some environments (CI/dev); skip tests that require it when missing
pytest.importorskip("sklearn")

from scripts import feature_engineering as fe


def setup_function():
    # reset module-level state between tests
    fe._encoder = None
    fe._scaler = None


def test_compute_haversine_zero_and_scale():
    # identical points -> zero distance
    lat = np.array([0.0, 10.0])
    lon = np.array([0.0, 20.0])
    d = fe.compute_haversine(lat, lon, lat, lon)
    assert np.allclose(d, 0)

    # one degree longitude at equator ~111 km
    d2 = fe.compute_haversine(0.0, 0.0, 0.0, 1.0)
    assert 111.0 < d2 < 112.0


def test_add_distance_column_fills_missing_with_median():
    df = pd.DataFrame({
        'pickup_latitude': [0.0, 0.0, np.nan],
        'pickup_longitude': [0.0, 1.0, 2.0],
        'dropoff_latitude': [0.0, 0.0, 0.0],
        'dropoff_longitude': [0.0, 1.0, 2.0],
    })

    out = fe.add_distance_column(df.copy())
    assert 'distance_km' in out.columns
    assert not out['distance_km'].isna().any()

    # last value (originally NaN) should equal median of first two distances
    d0 = fe.compute_haversine(0.0, 0.0, 0.0, 0.0)
    d1 = fe.compute_haversine(0.0, 1.0, 0.0, 1.0)
    expected_median = np.median([d0, d1])
    assert out['distance_km'].iloc[2] == pytest.approx(expected_median)


def test_enrich_datetime_parsing_and_fill():
    df = pd.DataFrame({'pickup_datetime': ['2020-01-01 00:00:00', 'not a date']})
    out = fe.enrich_datetime(df.copy())

    for col in ['pickup_hour', 'pickup_day', 'pickup_weekday', 'pickup_month']:
        assert col in out.columns

    # invalid date should be coerced and then filled with median (i.e., same as the valid row)
    assert out.at[0, 'pickup_hour'] == out.at[1, 'pickup_hour'] == 0
    assert pd.api.types.is_datetime64_any_dtype(out['pickup_datetime'])


def test_transform_categoricals_fit_and_transform_consistency():
    # Fit on a DataFrame with a missing value to exercise the Unknown handling
    df = pd.DataFrame({
        'vendor_id': ['V1', None],
        'store_and_fwd_flag': ['Y', 'N']
    })

    fe._encoder = None
    out = fe.transform_categoricals(df.copy(), fit=True)

    # original category columns should be removed
    assert 'vendor_id' not in out.columns
    assert 'store_and_fwd_flag' not in out.columns

    # one-hot columns created for categories
    assert any(c.startswith('vendor_id_') for c in out.columns)
    assert any(c.startswith('store_and_fwd_flag_') for c in out.columns)

    # Transform a new dataframe (no fitting) using the existing encoder
    new = pd.DataFrame({'vendor_id': ['V1'], 'store_and_fwd_flag': ['N']})
    out2 = fe.transform_categoricals(new.copy(), fit=False)
    assert list(out2.columns) == list(out.columns)


def test_normalize_numeric_fit_and_reuse():
    fe._scaler = None
    df = pd.DataFrame({
        'distance_km': [1.0, 2.0, 3.0],
        'pickup_hour': [0, 12, 23],
        'pickup_day': [1, 2, 3],
        'pickup_weekday': [1, 2, 3],
        'pickup_month': [1, 1, 1],
        'passenger_count': [1, 2, 3]
    })

    out = fe.normalize_numeric(df.copy(), fit=True)

    # after fitting, columns should have zero mean and unit std (population std dev)
    for col in ['distance_km', 'pickup_hour', 'pickup_day', 'pickup_weekday', 'pickup_month', 'passenger_count']:
        col_vals = out[col].values
        assert abs(col_vals.mean()) < 1e-6
        std = np.std(col_vals, ddof=0)
        # If the original column was constant (e.g., pickup_month here), StandardScaler yields zeros (std == 0)
        if np.allclose(col_vals, 0):
            assert std == pytest.approx(0.0)
        else:
            assert abs(std - 1.0) < 1e-6

    # Reuse scaler on a new dataframe
    new = df.copy()
    out2 = fe.normalize_numeric(new.copy(), fit=False)
    # values transformed should be finite and not all equal
    assert np.isfinite(out2['distance_km']).all()


def test_feature_engineering_pipeline_end_to_end():
    fe._encoder = None
    fe._scaler = None

    df = pd.DataFrame({
        'id': [1, 2],
        'vendor_id': ['V1', 'V2'],
        'store_and_fwd_flag': ['Y', 'N'],
        'pickup_latitude': [0.0, 0.0],
        'pickup_longitude': [0.0, 1.0],
        'dropoff_latitude': [0.0, 0.0],
        'dropoff_longitude': [0.0, 1.0],
        'pickup_datetime': ['2020-01-01 00:00:00', '2020-01-02 01:00:00'],
        'dropoff_datetime': ['2020-01-01 00:10:00', '2020-01-02 01:10:00'],
        'trip_duration': [600, 700],
        'passenger_count': [1, 2]
    })

    X, y, cols = fe.feature_engineering(df.copy(), fit=True, save=False, is_train=True)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(cols, list)
    assert cols == X.columns.tolist()

    # trip_duration removed from X and returned as y
    assert 'trip_duration' not in X.columns
    assert y.tolist() == [600, 700]

    # categorical encoded columns are present
    assert any(c.startswith('vendor_id_') for c in cols)


if __name__ == '__main__':
    pytest.main([__file__])
