"""
Unit tests for Time Series Analysis Project.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config_manager import Config
from data_utils import DataGenerator, DataPreprocessor, save_data, load_data
from models import ProphetForecaster, ARIMAForecaster, LSTMForecaster, AnomalyDetector
from visualization import TimeSeriesVisualizer


class TestDataGenerator:
    """Test cases for DataGenerator class."""
    
    def setup_method(self):
        """Setup test data."""
        self.config = {
            'synthetic': {
                'periods': 100,
                'trend_start': 10,
                'trend_end': 20,
                'seasonal_amplitude': 2,
                'noise_std': 1.0,
                'seed': 42
            }
        }
        self.generator = DataGenerator(self.config)
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        data = self.generator.generate_synthetic_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert 'ds' in data.columns
        assert 'y' in data.columns
        assert isinstance(data['ds'].iloc[0], pd.Timestamp)
        assert isinstance(data['y'].iloc[0], (int, float))
    
    def test_add_anomalies(self):
        """Test anomaly addition."""
        data = self.generator.generate_synthetic_data()
        data_with_anomalies = self.generator.add_anomalies(data, anomaly_rate=0.1)
        
        assert 'is_anomaly' in data_with_anomalies.columns
        assert data_with_anomalies['is_anomaly'].sum() > 0
        assert len(data_with_anomalies) == len(data)


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def setup_method(self):
        """Setup test data."""
        self.preprocessor = DataPreprocessor()
        
        # Create test data
        dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
        values = np.random.randn(200).cumsum() + 100
        self.data = pd.DataFrame({'ds': dates, 'y': values})
    
    def test_prepare_data_for_lstm(self):
        """Test LSTM data preparation."""
        X, y = self.preprocessor.prepare_data_for_lstm(self.data, sequence_length=10)
        
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 10  # sequence_length
        assert X.shape[2] == 1   # features
        assert len(y) == len(self.data) - 10
    
    def test_split_data(self):
        """Test data splitting."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y, test_size=0.2)
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    def test_scale_data(self):
        """Test data scaling."""
        data = np.random.randn(100, 2)
        
        scaled_data = self.preprocessor.scale_data(data, method='standard')
        
        assert scaled_data.shape == data.shape
        assert np.isclose(scaled_data.mean(axis=0), 0).all()
        assert np.isclose(scaled_data.std(axis=0), 1).all()
    
    def test_inverse_scale_data(self):
        """Test inverse scaling."""
        original_data = np.random.randn(100, 2)
        
        # Scale data
        scaled_data = self.preprocessor.scale_data(original_data, method='standard')
        
        # Inverse scale
        inverse_scaled = self.preprocessor.inverse_scale_data(scaled_data)
        
        assert np.allclose(original_data, inverse_scaled, atol=1e-10)


class TestProphetForecaster:
    """Test cases for ProphetForecaster class."""
    
    def setup_method(self):
        """Setup test data."""
        self.config = {
            'params': {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0
            }
        }
        self.forecaster = ProphetForecaster(self.config)
        
        # Create test data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        values = np.random.randn(100).cumsum() + 100
        self.data = pd.DataFrame({'ds': dates, 'y': values})
    
    def test_fit(self):
        """Test model fitting."""
        self.forecaster.fit(self.data)
        
        assert self.forecaster.is_fitted
        assert self.forecaster.model is not None
    
    def test_predict(self):
        """Test prediction."""
        self.forecaster.fit(self.data)
        forecast = self.forecaster.predict(30)
        
        assert isinstance(forecast, pd.DataFrame)
        assert 'ds' in forecast.columns
        assert 'yhat' in forecast.columns
        assert len(forecast) == 130  # 100 historical + 30 forecast
    
    def test_get_confidence_intervals(self):
        """Test confidence intervals."""
        self.forecaster.fit(self.data)
        intervals = self.forecaster.get_confidence_intervals(30)
        
        assert isinstance(intervals, dict)
        assert '80%' in intervals
        assert '95%' in intervals


class TestARIMAForecaster:
    """Test cases for ARIMAForecaster class."""
    
    def setup_method(self):
        """Setup test data."""
        self.config = {
            'auto_arima': True,
            'order': [1, 1, 1],
            'seasonal_order': [1, 1, 1, 12]
        }
        self.forecaster = ARIMAForecaster(self.config)
        
        # Create test data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        values = np.random.randn(100).cumsum() + 100
        self.data = pd.DataFrame({'ds': dates, 'y': values})
    
    def test_fit(self):
        """Test model fitting."""
        self.forecaster.fit(self.data)
        
        assert self.forecaster.is_fitted
        assert self.forecaster.model is not None
    
    def test_predict(self):
        """Test prediction."""
        self.forecaster.fit(self.data)
        forecast = self.forecaster.predict(30)
        
        assert isinstance(forecast, pd.DataFrame)
        assert 'ds' in forecast.columns
        assert 'yhat' in forecast.columns
        assert len(forecast) == 30


class TestAnomalyDetector:
    """Test cases for AnomalyDetector class."""
    
    def setup_method(self):
        """Setup test data."""
        self.config = {
            'isolation_forest': {
                'contamination': 0.1
            }
        }
        self.detector = AnomalyDetector(self.config)
        
        # Create test data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        values = np.random.randn(100).cumsum() + 100
        self.data = pd.DataFrame({'ds': dates, 'y': values})
    
    def test_detect_with_isolation_forest(self):
        """Test Isolation Forest anomaly detection."""
        result = self.detector.detect_with_isolation_forest(self.data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'is_anomaly_if' in result.columns
        assert len(result) == len(self.data)
    
    def test_detect_with_statistical_method(self):
        """Test statistical anomaly detection."""
        result = self.detector.detect_with_statistical_method(self.data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'is_anomaly_stat' in result.columns
        assert len(result) == len(self.data)


class TestConfigManager:
    """Test cases for Config class."""
    
    def setup_method(self):
        """Setup test configuration."""
        self.test_config = {
            'data': {
                'synthetic': {
                    'periods': 100,
                    'trend_start': 10
                }
            },
            'models': {
                'prophet': {
                    'enable': True,
                    'params': {
                        'changepoint_prior_scale': 0.05
                    }
                }
            }
        }
    
    def test_get_config_value(self):
        """Test getting configuration values."""
        config = Config()
        config._config = self.test_config
        
        assert config.get('data.synthetic.periods') == 100
        assert config.get('models.prophet.enable') is True
        assert config.get('nonexistent.key', 'default') == 'default'
    
    def test_get_model_config(self):
        """Test getting model configuration."""
        config = Config()
        config._config = self.test_config
        
        prophet_config = config.get_model_config('prophet')
        assert prophet_config['enable'] is True
        assert prophet_config['params']['changepoint_prior_scale'] == 0.05
    
    def test_is_model_enabled(self):
        """Test checking if model is enabled."""
        config = Config()
        config._config = self.test_config
        
        assert config.is_model_enabled('prophet') is True
        assert config.is_model_enabled('nonexistent') is False


class TestVisualization:
    """Test cases for visualization functions."""
    
    def setup_method(self):
        """Setup test data."""
        self.config = {
            'plot_style': 'default',
            'figure_size': [10, 6],
            'dpi': 100,
            'colors': {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'anomaly': '#d62728'
            }
        }
        self.visualizer = TimeSeriesVisualizer(self.config)
        
        # Create test data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        values = np.random.randn(100).cumsum() + 100
        self.data = pd.DataFrame({'ds': dates, 'y': values})
    
    def test_plot_time_series(self):
        """Test time series plotting."""
        fig = self.visualizer.plot_time_series(self.data, "Test Plot")
        
        assert fig is not None
        assert hasattr(fig, 'axes')
    
    def test_plot_anomalies(self):
        """Test anomaly plotting."""
        # Add anomaly column
        data_with_anomalies = self.data.copy()
        data_with_anomalies['is_anomaly'] = False
        data_with_anomalies.loc[10:15, 'is_anomaly'] = True
        
        fig = self.visualizer.plot_anomalies(data_with_anomalies, "Test Anomalies")
        
        assert fig is not None
        assert hasattr(fig, 'axes')


def test_save_load_data():
    """Test data saving and loading."""
    # Create test data
    dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
    values = np.random.randn(10)
    test_data = pd.DataFrame({'ds': dates, 'y': values})
    
    # Test CSV save/load
    save_data(test_data, 'test_data.csv')
    loaded_data = load_data('test_data.csv')
    
    assert isinstance(loaded_data, pd.DataFrame)
    assert len(loaded_data) == len(test_data)
    assert 'ds' in loaded_data.columns
    assert 'y' in loaded_data.columns
    
    # Clean up
    Path('test_data.csv').unlink()


if __name__ == "__main__":
    pytest.main([__file__])
