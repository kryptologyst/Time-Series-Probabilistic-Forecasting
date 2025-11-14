"""
Data generation and preprocessing module for Time Series Analysis Project.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataGenerator:
    """Generate synthetic time series data for analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data generator.
        
        Args:
            config: Configuration dictionary for data generation
        """
        self.config = config
        self.synthetic_config = config.get('synthetic', {})
    
    def generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic time series data with trend, seasonality, and noise.
        
        Returns:
            DataFrame with 'ds' (datetime) and 'y' (values) columns
        """
        periods = self.synthetic_config.get('periods', 1000)
        trend_start = self.synthetic_config.get('trend_start', 10)
        trend_end = self.synthetic_config.get('trend_end', 50)
        seasonal_amplitude = self.synthetic_config.get('seasonal_amplitude', 5)
        noise_std = self.synthetic_config.get('noise_std', 2.0)
        seed = self.synthetic_config.get('seed', 42)
        
        np.random.seed(seed)
        
        # Generate time index
        t = pd.date_range(start="2020-01-01", periods=periods, freq='D')
        
        # Generate trend component
        trend = np.linspace(trend_start, trend_end, periods)
        
        # Generate seasonal component (annual seasonality)
        seasonal = seasonal_amplitude * np.sin(2 * np.pi * t.dayofyear / 365)
        
        # Add weekly seasonality
        weekly_seasonal = 2 * np.sin(2 * np.pi * t.dayofweek / 7)
        
        # Generate noise
        noise = np.random.normal(0, noise_std, periods)
        
        # Combine components
        y = trend + seasonal + weekly_seasonal + noise
        
        df = pd.DataFrame({'ds': t, 'y': y})
        
        logger.info(f"Generated synthetic data with {periods} periods")
        return df
    
    def add_anomalies(self, df: pd.DataFrame, anomaly_rate: float = 0.05) -> pd.DataFrame:
        """
        Add anomalies to the time series data.
        
        Args:
            df: Input DataFrame
            anomaly_rate: Proportion of data points to make anomalous
            
        Returns:
            DataFrame with anomalies added
        """
        df_anomaly = df.copy()
        n_anomalies = int(len(df) * anomaly_rate)
        
        # Randomly select indices for anomalies
        anomaly_indices = np.random.choice(len(df), n_anomalies, replace=False)
        
        # Add anomalies (spikes or drops)
        for idx in anomaly_indices:
            if np.random.random() > 0.5:
                # Positive spike
                df_anomaly.loc[idx, 'y'] += np.random.uniform(10, 20)
            else:
                # Negative spike
                df_anomaly.loc[idx, 'y'] -= np.random.uniform(10, 20)
        
        df_anomaly['is_anomaly'] = False
        df_anomaly.loc[anomaly_indices, 'is_anomaly'] = True
        
        logger.info(f"Added {n_anomalies} anomalies to the data")
        return df_anomaly


class DataPreprocessor:
    """Preprocess time series data for modeling."""
    
    def __init__(self):
        """Initialize data preprocessor."""
        self.scaler = None
        self.is_fitted = False
    
    def prepare_data_for_lstm(self, df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model.
        
        Args:
            df: Input DataFrame with 'y' column
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) arrays for LSTM training
        """
        values = df['y'].values
        
        X, y = [], []
        for i in range(sequence_length, len(values)):
            X.append(values[i-sequence_length:i])
            y.append(values[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        logger.info(f"Prepared LSTM data: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.
        
        Args:
            X: Input features
            y: Target values
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        logger.info(f"Split data: train {X_train.shape[0]} samples, test {X_test.shape[0]} samples")
        return X_train, X_test, y_train, y_test
    
    def scale_data(self, data: np.ndarray, method: str = 'standard') -> np.ndarray:
        """
        Scale data using specified method.
        
        Args:
            data: Input data array
            method: Scaling method ('standard' or 'minmax')
            
        Returns:
            Scaled data array
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        if not self.is_fitted:
            scaled_data = scaler.fit_transform(data)
            self.scaler = scaler
            self.is_fitted = True
        else:
            scaled_data = self.scaler.transform(data)
        
        logger.info(f"Scaled data using {method} scaling")
        return scaled_data
    
    def inverse_scale_data(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data.
        
        Args:
            data: Scaled data array
            
        Returns:
            Original scale data array
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call scale_data first.")
        
        return self.scaler.inverse_transform(data)


def load_external_data(url: str) -> pd.DataFrame:
    """
    Load external time series data from URL.
    
    Args:
        url: URL to the data file
        
    Returns:
        DataFrame with loaded data
    """
    try:
        df = pd.read_csv(url)
        logger.info(f"Loaded external data from {url}")
        return df
    except Exception as e:
        logger.error(f"Failed to load external data from {url}: {e}")
        raise


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save DataFrame to file.
    
    Args:
        df: DataFrame to save
        filepath: Path to save the file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.endswith('.csv'):
        df.to_csv(filepath, index=False)
    elif filepath.endswith('.parquet'):
        df.to_parquet(filepath, index=False)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    logger.info(f"Saved data to {filepath}")


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load DataFrame from file.
    
    Args:
        filepath: Path to the data file
        
    Returns:
        Loaded DataFrame
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    logger.info(f"Loaded data from {filepath}")
    return df
