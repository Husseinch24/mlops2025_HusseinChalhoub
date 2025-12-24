import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

pytest.importorskip("joblib")

from src.ml_project.inference.inference import BatchPredictor


class TestBatchPredictorInit:
    """Test BatchPredictor initialization"""
    
    def test_init_with_model(self):
        """Test initialization with a model"""
        mock_model = Mock()
        predictor = BatchPredictor(model=mock_model)
        assert predictor.model is mock_model
    
    def test_init_without_model(self):
        """Test initialization without a model (default None)"""
        predictor = BatchPredictor()
        assert predictor.model is None


class TestFeatureEngineering:
    """Test feature engineering static method"""
    
    def test_feature_engineering_with_target(self):
        """Test feature engineering with target column"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        
        X, y, feature_cols = BatchPredictor.feature_engineering(df, fit=False, is_train=True)
        
        assert isinstance(X, pd.DataFrame)
        assert len(X) == 3
        assert set(feature_cols) == {'feature1', 'feature2'}
        assert y is not None
        assert len(y) == 3
        np.testing.assert_array_equal(y.values, [0, 1, 0])
    
    def test_feature_engineering_without_target(self):
        """Test feature engineering without target column"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        X, y, feature_cols = BatchPredictor.feature_engineering(df, fit=False, is_train=False)
        
        assert isinstance(X, pd.DataFrame)
        assert len(X) == 3
        assert set(feature_cols) == {'feature1', 'feature2'}
        assert y is None
    
    def test_feature_engineering_returns_copy(self):
        """Test that feature engineering returns a copy of features"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'target': [0, 1, 0]
        })
        
        X, _, _ = BatchPredictor.feature_engineering(df)
        X.loc[0, 'feature1'] = 999
        
        # Original dataframe should not be affected
        assert df.loc[0, 'feature1'] == 1


class TestBatchInference:
    """Test batch inference functionality"""
    
    def test_batch_inference_without_model(self):
        """Test batch inference raises error when model is not loaded"""
        predictor = BatchPredictor(model=None)
        df = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Model is not loaded"):
            predictor.batch_inference(df)
    
    def test_batch_inference_basic(self):
        """Test basic batch inference"""
        # Create a mock model that returns predictions
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0.5, 1.5, 2.5]))
        
        predictor = BatchPredictor(model=mock_model)
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 2]
        })
        
        result = predictor.batch_inference(df)
        
        assert 'prediction' in result.columns
        assert 'timestamp' in result.columns
        assert len(result) == 3
        np.testing.assert_array_equal(result['prediction'].values, [0.5, 1.5, 2.5])
    
    def test_batch_inference_adds_timestamp(self):
        """Test that batch inference adds timestamp column"""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([1, 1, 1]))
        
        predictor = BatchPredictor(model=mock_model)
        df = pd.DataFrame({'feature1': [1, 2, 3]})
        
        before_inference = datetime.now()
        result = predictor.batch_inference(df)
        after_inference = datetime.now()
        
        assert 'timestamp' in result.columns
        assert all(before_inference <= ts <= after_inference for ts in result['timestamp'])
    
    def test_batch_inference_save_to_directory(self):
        """Test saving predictions to a directory"""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([1, 2, 3]))
        
        predictor = BatchPredictor(model=mock_model)
        df = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = predictor.batch_inference(df, save_path=tmpdir)
            
            # Check that file was created
            files = os.listdir(tmpdir)
            assert len(files) == 1
            assert files[0].endswith('_predictions.csv')
            
            # Verify saved file content
            saved_df = pd.read_csv(os.path.join(tmpdir, files[0]))
            assert 'prediction' in saved_df.columns
            assert len(saved_df) == 3
    
    def test_batch_inference_save_to_file(self):
        """Test saving predictions to a specific file path"""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([1, 2, 3]))
        
        predictor = BatchPredictor(model=mock_model)
        df = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_file = os.path.join(tmpdir, 'my_predictions.csv')
            result = predictor.batch_inference(df, save_path=save_file)
            
            # Check that file was created with the correct name
            assert os.path.exists(save_file)
            
            # Verify saved file content
            saved_df = pd.read_csv(save_file)
            assert 'prediction' in saved_df.columns
            assert len(saved_df) == 3
    
    def test_batch_inference_without_save(self):
        """Test batch inference without saving"""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([1, 2, 3]))
        
        predictor = BatchPredictor(model=mock_model)
        df = pd.DataFrame({'feature1': [1, 2, 3]})
        
        result = predictor.batch_inference(df, save_path=None)
        
        assert 'prediction' in result.columns
        assert len(result) == 3


class TestLoadModel:
    """Test model loading functionality"""
    
    def test_load_model_default_path(self):
        """Test loading model with default path"""
        with patch('joblib.load') as mock_load:
            mock_load.return_value = Mock()
            model = BatchPredictor.load_model()
            mock_load.assert_called_once_with('best_model.pkl')
    
    def test_load_model_custom_path(self):
        """Test loading model with custom path"""
        with patch('joblib.load') as mock_load:
            mock_load.return_value = Mock()
            custom_path = '/path/to/custom_model.pkl'
            model = BatchPredictor.load_model(filepath=custom_path)
            mock_load.assert_called_once_with(custom_path)
    
    def test_load_model_returns_model(self):
        """Test that load_model returns the loaded model"""
        with patch('joblib.load') as mock_load:
            expected_model = Mock(name='MockModel')
            mock_load.return_value = expected_model
            model = BatchPredictor.load_model()
            assert model is expected_model


class TestBatchPredictorIntegration:
    """Integration tests for BatchPredictor"""
    
    def test_full_workflow(self):
        """Test complete workflow from initialization to inference"""
        # Create a simple mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0.8, 0.3, 0.9, 0.2]))
        
        # Initialize predictor with model
        predictor = BatchPredictor(model=mock_model)
        
        # Create test data
        test_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [5.0, 6.0, 7.0, 8.0],
            'feature3': [9.0, 10.0, 11.0, 12.0]
        })
        
        # Run inference
        with tempfile.TemporaryDirectory() as tmpdir:
            result = predictor.batch_inference(test_data, save_path=tmpdir)
            
            # Verify results
            assert len(result) == 4
            assert 'prediction' in result.columns
            assert 'timestamp' in result.columns
            np.testing.assert_array_equal(result['prediction'].values, [0.8, 0.3, 0.9, 0.2])
            
            # Verify file was saved
            files = os.listdir(tmpdir)
            assert len(files) == 1
