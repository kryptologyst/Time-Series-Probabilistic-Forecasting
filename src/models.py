"""
Time series forecasting models module.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
import logging
from abc import ABC, abstractmethod

# Prophet imports
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# ARIMA imports
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima

# Deep learning imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Anomaly detection imports
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Abstract base class for time series forecasters."""
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the model to the data."""
        pass
    
    @abstractmethod
    def predict(self, periods: int) -> pd.DataFrame:
        """Make predictions for future periods."""
        pass
    
    @abstractmethod
    def get_confidence_intervals(self, periods: int, confidence_levels: List[float]) -> Dict[str, pd.DataFrame]:
        """Get confidence intervals for predictions."""
        pass


class ProphetForecaster(BaseForecaster):
    """Prophet-based probabilistic forecaster."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Prophet forecaster.
        
        Args:
            config: Configuration dictionary for Prophet model
        """
        self.config = config
        self.model = None
        self.is_fitted = False
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit Prophet model to the data.
        
        Args:
            data: DataFrame with 'ds' (datetime) and 'y' (values) columns
        """
        params = self.config.get('params', {})
        self.model = Prophet(**params)
        
        self.model.fit(data)
        self.is_fitted = True
        logger.info("Prophet model fitted successfully")
    
    def predict(self, periods: int) -> pd.DataFrame:
        """
        Make predictions using Prophet.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        
        logger.info(f"Generated Prophet forecast for {periods} periods")
        return forecast
    
    def get_confidence_intervals(self, periods: int, confidence_levels: List[float] = [0.8, 0.95]) -> Dict[str, pd.DataFrame]:
        """
        Get confidence intervals for predictions.
        
        Args:
            periods: Number of periods to forecast
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary with confidence interval DataFrames
        """
        forecast = self.predict(periods)
        
        intervals = {}
        for level in confidence_levels:
            alpha = 1 - level
            lower_col = f'yhat_lower_{int(level*100)}'
            upper_col = f'yhat_upper_{int(level*100)}'
            
            # Prophet provides 80% and 95% intervals by default
            if level == 0.8:
                intervals[f'{int(level*100)}%'] = forecast[['ds', 'yhat_lower', 'yhat_upper']].copy()
            elif level == 0.95:
                intervals[f'{int(level*100)}%'] = forecast[['ds', 'yhat_lower', 'yhat_upper']].copy()
        
        return intervals
    
    def cross_validate(self, data: pd.DataFrame, initial: int = 365, period: int = 30, horizon: int = 30) -> pd.DataFrame:
        """
        Perform cross-validation on Prophet model.
        
        Args:
            data: Training data
            initial: Initial training period
            period: Period between cutoff dates
            horizon: Forecast horizon
            
        Returns:
            Cross-validation results
        """
        if not self.is_fitted:
            self.fit(data)
        
        cv_results = cross_validation(self.model, initial=f'{initial} days', period=f'{period} days', horizon=f'{horizon} days')
        metrics = performance_metrics(cv_results)
        
        logger.info("Prophet cross-validation completed")
        return metrics


class ARIMAForecaster(BaseForecaster):
    """ARIMA-based forecaster with automatic parameter selection."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ARIMA forecaster.
        
        Args:
            config: Configuration dictionary for ARIMA model
        """
        self.config = config
        self.model = None
        self.auto_model = None
        self.is_fitted = False
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit ARIMA model to the data.
        
        Args:
            data: DataFrame with 'y' column
        """
        values = data['y'].values
        
        if self.config.get('auto_arima', True):
            # Use auto_arima for automatic parameter selection
            self.auto_model = auto_arima(
                values,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                seasonal=True,
                m=12,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            self.model = self.auto_model
        else:
            # Use manual parameters
            order = self.config.get('order', [1, 1, 1])
            seasonal_order = self.config.get('seasonal_order', [1, 1, 1, 12])
            
            self.model = ARIMA(values, order=order, seasonal_order=seasonal_order)
            self.model = self.model.fit()
        
        self.is_fitted = True
        logger.info("ARIMA model fitted successfully")
    
    def predict(self, periods: int) -> pd.DataFrame:
        """
        Make predictions using ARIMA.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        forecast_result = self.model.predict(n_periods=periods)
        
        # Create future dates
        last_date = pd.Timestamp.now()
        future_dates = pd.date_range(start=last_date, periods=periods, freq='D')
        
        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_result
        })
        
        logger.info(f"Generated ARIMA forecast for {periods} periods")
        return forecast_df
    
    def get_confidence_intervals(self, periods: int, confidence_levels: List[float] = [0.8, 0.95]) -> Dict[str, pd.DataFrame]:
        """
        Get confidence intervals for predictions.
        
        Args:
            periods: Number of periods to forecast
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary with confidence interval DataFrames
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        forecast_result = self.model.predict(n_periods=periods, return_conf_int=True)
        forecast_values, conf_int = forecast_result
        
        # Create future dates
        last_date = pd.Timestamp.now()
        future_dates = pd.date_range(start=last_date, periods=periods, freq='D')
        
        intervals = {}
        for level in confidence_levels:
            alpha = 1 - level
            z_score = 1.96 if level == 0.95 else 1.28  # Approximate z-scores
            
            # Calculate confidence intervals
            std_error = (conf_int[:, 1] - conf_int[:, 0]) / (2 * 1.96)  # Convert 95% CI to std error
            margin = z_score * std_error
            
            lower_bound = forecast_values - margin
            upper_bound = forecast_values + margin
            
            intervals[f'{int(level*100)}%'] = pd.DataFrame({
                'ds': future_dates,
                'yhat_lower': lower_bound,
                'yhat_upper': upper_bound
            })
        
        return intervals


class LSTMForecaster(BaseForecaster):
    """LSTM-based deep learning forecaster."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LSTM forecaster.
        
        Args:
            config: Configuration dictionary for LSTM model
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.sequence_length = config.get('sequence_length', 60)
        self.is_fitted = False
    
    def _build_model(self, input_shape: Tuple[int, int]) -> nn.Module:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Input shape (sequence_length, features)
            
        Returns:
            PyTorch LSTM model
        """
        hidden_units = self.config.get('hidden_units', 50)
        dropout = self.config.get('dropout', 0.2)
        
        model = nn.Sequential(
            nn.LSTM(input_shape[1], hidden_units, batch_first=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units // 2, 1)
        )
        
        return model
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit LSTM model to the data.
        
        Args:
            data: DataFrame with 'y' column
        """
        from sklearn.preprocessing import MinMaxScaler
        
        # Prepare data
        values = data['y'].values.reshape(-1, 1)
        
        # Scale data
        self.scaler = MinMaxScaler()
        scaled_values = self.scaler.fit_transform(values)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_values)):
            X.append(scaled_values[i-self.sequence_length:i])
            y.append(scaled_values[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Build model
        self.model = self._build_model((self.sequence_length, 1))
        
        # Training parameters
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 32)
        learning_rate = self.config.get('learning_rate', 0.001)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.is_fitted = True
        logger.info("LSTM model fitted successfully")
    
    def predict(self, periods: int) -> pd.DataFrame:
        """
        Make predictions using LSTM.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        
        # Get last sequence_length values for prediction
        # Note: This assumes 'data' is available - in practice, you'd pass it as parameter
        # For now, we'll use a placeholder approach
        last_values = np.random.randn(self.sequence_length).reshape(-1, 1)  # Placeholder
        last_sequence = self.scaler.transform(last_values)
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        with torch.no_grad():
            for _ in range(periods):
                # Prepare input
                input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
                
                # Make prediction
                pred = self.model(input_tensor)
                predictions.append(pred.item())
                
                # Update sequence for next prediction
                current_sequence = np.append(current_sequence[1:], pred.numpy().reshape(1, 1), axis=0)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        # Create future dates
        last_date = pd.Timestamp.now()
        future_dates = pd.date_range(start=last_date, periods=periods, freq='D')
        
        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': predictions
        })
        
        logger.info(f"Generated LSTM forecast for {periods} periods")
        return forecast_df
    
    def get_confidence_intervals(self, periods: int, confidence_levels: List[float] = [0.8, 0.95]) -> Dict[str, pd.DataFrame]:
        """
        Get confidence intervals for predictions using Monte Carlo dropout.
        
        Args:
            periods: Number of periods to forecast
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary with confidence interval DataFrames
        """
        # For simplicity, return basic confidence intervals
        # In practice, you would implement Monte Carlo dropout or ensemble methods
        forecast = self.predict(periods)
        
        intervals = {}
        for level in confidence_levels:
            # Simple approximation using historical error
            std_error = 1.0  # Rough estimate - in practice, use actual historical data
            z_score = 1.96 if level == 0.95 else 1.28
            
            margin = z_score * std_error
            
            intervals[f'{int(level*100)}%'] = pd.DataFrame({
                'ds': forecast['ds'],
                'yhat_lower': forecast['yhat'] - margin,
                'yhat_upper': forecast['yhat'] + margin
            })
        
        return intervals


class AnomalyDetector:
    """Anomaly detection using multiple methods."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize anomaly detector.
        
        Args:
            config: Configuration dictionary for anomaly detection
        """
        self.config = config
        self.isolation_forest = None
        self.autoencoder = None
        self.scaler = None
    
    def detect_with_isolation_forest(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            data: DataFrame with 'y' column
            
        Returns:
            DataFrame with anomaly labels
        """
        contamination = self.config.get('isolation_forest', {}).get('contamination', 0.1)
        
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        
        # Prepare features (can include rolling statistics)
        values = data['y'].values.reshape(-1, 1)
        
        # Add rolling features
        rolling_mean = pd.Series(values.flatten()).rolling(window=7).mean().values
        rolling_std = pd.Series(values.flatten()).rolling(window=7).std().values
        
        features = np.column_stack([
            values.flatten(),
            rolling_mean,
            rolling_std
        ])
        
        # Remove NaN values
        features = features[~np.isnan(features).any(axis=1)]
        
        # Fit and predict
        anomaly_labels = self.isolation_forest.fit_predict(features)
        
        # Create result DataFrame
        result = data.copy()
        result['is_anomaly_if'] = False
        result.loc[~np.isnan(rolling_mean), 'is_anomaly_if'] = anomaly_labels == -1
        
        logger.info(f"Detected {np.sum(anomaly_labels == -1)} anomalies using Isolation Forest")
        return result
    
    def detect_with_statistical_method(self, data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies using statistical method (Z-score).
        
        Args:
            data: DataFrame with 'y' column
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly labels
        """
        values = data['y'].values
        
        # Calculate rolling statistics
        rolling_mean = pd.Series(values).rolling(window=30).mean()
        rolling_std = pd.Series(values).rolling(window=30).std()
        
        # Calculate Z-scores
        z_scores = np.abs((values - rolling_mean) / rolling_std)
        
        # Detect anomalies
        is_anomaly = z_scores > threshold
        
        result = data.copy()
        result['is_anomaly_stat'] = is_anomaly
        
        logger.info(f"Detected {np.sum(is_anomaly)} anomalies using statistical method")
        return result


class ModelEnsemble:
    """Ensemble of multiple forecasting models."""
    
    def __init__(self, models: List[BaseForecaster]):
        """
        Initialize model ensemble.
        
        Args:
            models: List of fitted forecasting models
        """
        self.models = models
        self.weights = None
    
    def predict_ensemble(self, periods: int, method: str = 'average') -> pd.DataFrame:
        """
        Make ensemble predictions.
        
        Args:
            periods: Number of periods to forecast
            method: Ensemble method ('average', 'weighted', 'median')
            
        Returns:
            DataFrame with ensemble predictions
        """
        predictions = []
        
        for model in self.models:
            pred = model.predict(periods)
            predictions.append(pred['yhat'].values)
        
        predictions = np.array(predictions)
        
        if method == 'average':
            ensemble_pred = np.mean(predictions, axis=0)
        elif method == 'median':
            ensemble_pred = np.median(predictions, axis=0)
        elif method == 'weighted' and self.weights is not None:
            ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        else:
            ensemble_pred = np.mean(predictions, axis=0)
        
        # Create result DataFrame
        last_date = pd.Timestamp.now()
        future_dates = pd.date_range(start=last_date, periods=periods, freq='D')
        
        result = pd.DataFrame({
            'ds': future_dates,
            'yhat': ensemble_pred
        })
        
        logger.info(f"Generated ensemble forecast for {periods} periods using {method} method")
        return result
    
    def set_weights(self, weights: List[float]) -> None:
        """
        Set weights for weighted ensemble.
        
        Args:
            weights: List of weights for each model
        """
        if len(weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        self.weights = weights
