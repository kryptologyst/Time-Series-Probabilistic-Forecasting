"""
Visualization module for Time Series Analysis Project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TimeSeriesVisualizer:
    """Comprehensive time series visualization class."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize visualizer with configuration.
        
        Args:
            config: Visualization configuration dictionary
        """
        self.config = config
        self._setup_style()
    
    def _setup_style(self) -> None:
        """Setup matplotlib and seaborn styles."""
        style = self.config.get('plot_style', 'seaborn-v0_8')
        plt.style.use(style)
        
        # Set default figure size and DPI
        plt.rcParams['figure.figsize'] = self.config.get('figure_size', [12, 8])
        plt.rcParams['figure.dpi'] = self.config.get('dpi', 100)
        
        # Set colors
        self.colors = self.config.get('colors', {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'anomaly': '#d62728'
        })
    
    def plot_time_series(self, data: pd.DataFrame, title: str = "Time Series Plot", 
                        figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot basic time series.
        
        Args:
            data: DataFrame with 'ds' and 'y' columns
            title: Plot title
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize or self.config.get('figure_size', [12, 8]))
        
        ax.plot(data['ds'], data['y'], color=self.colors['primary'], linewidth=1.5)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        logger.info(f"Created time series plot: {title}")
        return fig
    
    def plot_forecast_with_intervals(self, historical_data: pd.DataFrame, 
                                   forecast: pd.DataFrame, 
                                   confidence_intervals: Optional[Dict[str, pd.DataFrame]] = None,
                                   title: str = "Forecast with Confidence Intervals",
                                   figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot forecast with confidence intervals.
        
        Args:
            historical_data: Historical data DataFrame
            forecast: Forecast DataFrame
            confidence_intervals: Dictionary of confidence interval DataFrames
            title: Plot title
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize or self.config.get('figure_size', [12, 8]))
        
        # Plot historical data
        ax.plot(historical_data['ds'], historical_data['y'], 
                color=self.colors['primary'], linewidth=1.5, label='Historical Data')
        
        # Plot forecast
        ax.plot(forecast['ds'], forecast['yhat'], 
                color=self.colors['secondary'], linewidth=2, label='Forecast')
        
        # Plot confidence intervals
        if confidence_intervals:
            for level, interval_df in confidence_intervals.items():
                alpha = 0.3 if level == '80%' else 0.2
                ax.fill_between(interval_df['ds'], 
                               interval_df['yhat_lower'], 
                               interval_df['yhat_upper'],
                               alpha=alpha, 
                               label=f'{level} Confidence Interval')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        logger.info(f"Created forecast plot with confidence intervals: {title}")
        return fig
    
    def plot_anomalies(self, data: pd.DataFrame, anomaly_column: str = 'is_anomaly',
                      title: str = "Anomaly Detection Results",
                      figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot time series with highlighted anomalies.
        
        Args:
            data: DataFrame with time series and anomaly labels
            anomaly_column: Column name containing anomaly labels
            title: Plot title
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize or self.config.get('figure_size', [12, 8]))
        
        # Plot normal data points
        normal_data = data[~data[anomaly_column]]
        ax.plot(normal_data['ds'], normal_data['y'], 
                color=self.colors['primary'], linewidth=1.5, label='Normal Data')
        
        # Plot anomalies
        anomaly_data = data[data[anomaly_column]]
        if not anomaly_data.empty:
            ax.scatter(anomaly_data['ds'], anomaly_data['y'], 
                      color=self.colors['anomaly'], s=50, 
                      label=f'Anomalies ({len(anomaly_data)})', zorder=5)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        logger.info(f"Created anomaly detection plot: {title}")
        return fig
    
    def plot_decomposition(self, data: pd.DataFrame, model: str = 'additive',
                          title: str = "Time Series Decomposition",
                          figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot time series decomposition.
        
        Args:
            data: DataFrame with 'ds' and 'y' columns
            model: Decomposition model ('additive' or 'multiplicative')
            title: Plot title
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Prepare data for decomposition
        ts_data = pd.Series(data['y'].values, index=data['ds'])
        
        # Perform decomposition
        decomposition = seasonal_decompose(ts_data, model=model, period=365)
        
        fig, axes = plt.subplots(4, 1, figsize=figsize or self.config.get('figure_size', [12, 10]))
        
        # Original series
        decomposition.observed.plot(ax=axes[0], color=self.colors['primary'])
        axes[0].set_title('Original Series')
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        decomposition.trend.plot(ax=axes[1], color=self.colors['secondary'])
        axes[1].set_title('Trend')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal
        decomposition.seasonal.plot(ax=axes[2], color='green')
        axes[2].set_title('Seasonal')
        axes[2].grid(True, alpha=0.3)
        
        # Residual
        decomposition.resid.plot(ax=axes[3], color='red')
        axes[3].set_title('Residual')
        axes[3].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        logger.info(f"Created decomposition plot: {title}")
        return fig
    
    def plot_model_comparison(self, forecasts: Dict[str, pd.DataFrame],
                            historical_data: pd.DataFrame,
                            title: str = "Model Comparison",
                            figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot comparison of multiple model forecasts.
        
        Args:
            forecasts: Dictionary of model forecasts
            historical_data: Historical data DataFrame
            title: Plot title
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize or self.config.get('figure_size', [12, 8]))
        
        # Plot historical data
        ax.plot(historical_data['ds'], historical_data['y'], 
                color='black', linewidth=2, label='Historical Data')
        
        # Plot forecasts
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            color = colors[i % len(colors)]
            ax.plot(forecast['ds'], forecast['yhat'], 
                    color=color, linewidth=2, label=f'{model_name} Forecast')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        logger.info(f"Created model comparison plot: {title}")
        return fig
    
    def create_interactive_plot(self, data: pd.DataFrame, 
                               forecast: Optional[pd.DataFrame] = None,
                               anomalies: Optional[pd.DataFrame] = None,
                               title: str = "Interactive Time Series Plot") -> go.Figure:
        """
        Create interactive Plotly plot.
        
        Args:
            data: Historical data DataFrame
            forecast: Optional forecast DataFrame
            anomalies: Optional anomalies DataFrame
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=data['ds'],
            y=data['y'],
            mode='lines',
            name='Historical Data',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        # Add forecast if provided
        if forecast is not None:
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color=self.colors['secondary'], width=2, dash='dash')
            ))
        
        # Add anomalies if provided
        if anomalies is not None and not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies['ds'],
                y=anomalies['y'],
                mode='markers',
                name='Anomalies',
                marker=dict(color=self.colors['anomaly'], size=8)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white'
        )
        
        logger.info(f"Created interactive plot: {title}")
        return fig
    
    def plot_residuals(self, actual: np.ndarray, predicted: np.ndarray,
                      title: str = "Residuals Analysis",
                      figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot residuals analysis.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            title: Plot title
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        residuals = actual - predicted
        
        fig, axes = plt.subplots(2, 2, figsize=figsize or self.config.get('figure_size', [12, 8]))
        
        # Residuals vs Fitted
        axes[0, 0].scatter(predicted, residuals, alpha=0.6, color=self.colors['primary'])
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_title('Residuals vs Fitted')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color=self.colors['secondary'])
        axes[1, 0].set_title('Histogram of Residuals')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals over time
        axes[1, 1].plot(residuals, color=self.colors['primary'])
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_title('Residuals over Time')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        logger.info(f"Created residuals analysis plot: {title}")
        return fig
    
    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300) -> None:
        """
        Save matplotlib figure to file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            dpi: DPI for saved image
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved plot to {filename}")
    
    def save_interactive_plot(self, fig: go.Figure, filename: str) -> None:
        """
        Save Plotly figure to HTML file.
        
        Args:
            fig: Plotly figure
            filename: Output filename
        """
        fig.write_html(filename)
        logger.info(f"Saved interactive plot to {filename}")


class MetricsVisualizer:
    """Visualizer for model performance metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metrics visualizer.
        
        Args:
            config: Visualization configuration dictionary
        """
        self.config = config
        self.colors = config.get('colors', {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'anomaly': '#d62728'
        })
    
    def plot_metrics_comparison(self, metrics: Dict[str, Dict[str, float]],
                               title: str = "Model Performance Comparison",
                               figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot comparison of model metrics.
        
        Args:
            metrics: Dictionary of model metrics
            title: Plot title
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        models = list(metrics.keys())
        metric_names = list(metrics[models[0]].keys())
        
        fig, axes = plt.subplots(2, 2, figsize=figsize or self.config.get('figure_size', [12, 8]))
        axes = axes.flatten()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, metric in enumerate(metric_names[:4]):  # Show first 4 metrics
            values = [metrics[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, color=colors[:len(models)])
            axes[i].set_title(f'{metric}')
            axes[i].set_ylabel('Value')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        logger.info(f"Created metrics comparison plot: {title}")
        return fig
