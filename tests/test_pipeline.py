import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.ml_project.pipelines.pipeline import TaxiPipeline
 
@pytest.fixture
def config(tmp_path):
    return {
        "train": {"metric": "mae", "test_size": 0.2, "seed": 42},
        "paths": {
            "train_csv": tmp_path / "train.csv",
            "test_csv": tmp_path / "test.csv",
            "artifact_dir": tmp_path / "artifacts",
            "output_dir": tmp_path / "outputs"
        }
    }
 
@pytest.fixture
def sample_csvs(config):
    # Create dummy train/test CSVs
    train_data = pd.DataFrame({
        "id": [1, 2],
        "pickup_latitude": [40.7, 40.8],
        "pickup_longitude": [-73.9, -73.95],
        "dropoff_latitude": [40.75, 40.82],
        "dropoff_longitude": [-73.98, -73.96],
        "pickup_datetime": ["2025-12-20 12:00:00", "2025-12-20 13:00:00"],
        "passenger_count": [1, 2],
        "trip_duration": [1800, 2700],
        "vendor_id": ["V1","V2"],
        "store_and_fwd_flag": ["N","Y"]
    })
    test_data = train_data.copy()
    train_data.to_csv(config["paths"]["train_csv"], index=False)
    test_data.to_csv(config["paths"]["test_csv"], index=False)
    return config
 
def test_pipeline_run(sample_csvs):
    cfg = sample_csvs
    pipeline = TaxiPipeline(cfg)
 
    # Patch heavy components
    with patch.object(pipeline, "preprocess", side_effect=lambda df, is_train=True: df), \
         patch.object(pipeline, "feature_engineering", return_value=(pd.DataFrame({"feature": [1,2]}), pd.Series([1800,2700]))), \
         patch.object(pipeline, "train", return_value=(MagicMock(spec=[]), "mock_model")), \
         patch.object(pipeline.model_trainer, "save_model", return_value="mock_path"):
 
        model, best_model_name, output_df = pipeline.run()
 
        # Assertions
        assert best_model_name == "mock_model"
        assert not output_df.empty
        assert "prediction" in output_df.columns
        assert "timestamp" in output_df.columns
 
 