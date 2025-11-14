# Time Series Probabilistic Forecasting

A comprehensive time series analysis and forecasting platform featuring probabilistic forecasting, anomaly detection, and multiple modeling approaches.

## Features

### Forecasting Models
- **Prophet**: Facebook's Prophet for robust forecasting with trend and seasonality
- **ARIMA**: Auto-ARIMA with automatic parameter selection
- **LSTM**: Deep learning approach using PyTorch for complex patterns
- **Ensemble Methods**: Combine multiple models for improved accuracy

### Anomaly Detection
- **Isolation Forest**: Unsupervised anomaly detection
- **Statistical Methods**: Z-score based anomaly detection
- **Visualization**: Interactive plots highlighting anomalies

### Probabilistic Forecasting
- **Confidence Intervals**: Multiple confidence levels (80%, 95%)
- **Uncertainty Quantification**: Capture prediction uncertainty
- **Risk Assessment**: Support for risk-sensitive decision making

### Data Sources
- **Synthetic Data**: Configurable synthetic time series with trends, seasonality, and noise
- **CSV Upload**: Support for custom datasets
- **External Datasets**: Integration with public time series datasets

### Visualization & Interface
- **Streamlit Dashboard**: Interactive web interface
- **Interactive Plots**: Plotly-based visualizations
- **Model Comparison**: Side-by-side forecast comparisons
- **Export Capabilities**: Download forecasts and results

## Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/kryptologyst/Time-Series-Probabilistic-Forecasting.git
   cd Time-Series-Probabilistic-Forecasting
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit dashboard**
   ```bash
   streamlit run app.py
   ```

4. **Or run from command line**
   ```bash
   python main.py --mode cli --models prophet arima --forecast-horizon 30
   ```

## Usage

### Streamlit Dashboard

Launch the interactive web interface:

```bash
streamlit run app.py
```

The dashboard provides:
- Data generation and upload
- Model training and comparison
- Interactive forecasting
- Anomaly detection
- Results visualization and export

### Command Line Interface

Run analysis from command line:

```bash
# Basic usage
python main.py --mode cli --models prophet arima

# With custom parameters
python main.py --mode cli \
    --models prophet arima lstm \
    --forecast-horizon 60 \
    --anomaly-detection \
    --output-dir results
```

### Programmatic Usage

```python
from src.config_manager import config
from src.data_utils import DataGenerator
from src.models import ProphetForecaster, ARIMAForecaster
from src.visualization import TimeSeriesVisualizer

# Generate synthetic data
generator = DataGenerator(config.get_data_config())
data = generator.generate_synthetic_data()

# Train Prophet model
prophet_config = config.get_model_config('prophet')
prophet_model = ProphetForecaster(prophet_config)
prophet_model.fit(data)

# Generate forecast
forecast = prophet_model.predict(30)
confidence_intervals = prophet_model.get_confidence_intervals(30)

# Visualize results
visualizer = TimeSeriesVisualizer(config.get_visualization_config())
fig = visualizer.plot_forecast_with_intervals(data, forecast, confidence_intervals)
```

## Configuration

The project uses YAML configuration files for easy customization:

### Main Configuration (`config/config.yaml`)

```yaml
data:
  synthetic:
    periods: 1000
    trend_start: 10
    trend_end: 50
    seasonal_amplitude: 5
    noise_std: 2.0

models:
  prophet:
    enable: true
    params:
      changepoint_prior_scale: 0.05
      seasonality_prior_scale: 10.0
  
  arima:
    enable: true
    params:
      auto_arima: true
      order: [1, 1, 1]

forecasting:
  horizon: 30
  confidence_intervals: [0.8, 0.95]
```

## Project Structure

```
time-series-analysis/
├── src/                    # Source code
│   ├── config_manager.py   # Configuration management
│   ├── data_utils.py       # Data generation and preprocessing
│   ├── models.py           # Forecasting models
│   └── visualization.py    # Visualization utilities
├── config/                 # Configuration files
│   └── config.yaml         # Main configuration
├── data/                   # Data storage
├── models/                 # Saved models
├── notebooks/              # Jupyter notebooks
├── tests/                  # Unit tests
├── app.py                  # Streamlit dashboard
├── main.py                 # Main CLI script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Models

### Prophet
- Robust to missing data and outliers
- Automatic trend and seasonality detection
- Built-in holiday effects
- Confidence intervals included

### ARIMA
- Auto-ARIMA for parameter selection
- Seasonal ARIMA support
- Statistical significance testing
- Residual analysis

### LSTM
- Deep learning approach
- Handles complex non-linear patterns
- Sequence-to-sequence prediction
- Configurable architecture

## Anomaly Detection

### Isolation Forest
- Unsupervised learning
- Handles high-dimensional data
- Robust to outliers
- Configurable contamination level

### Statistical Methods
- Z-score based detection
- Rolling window statistics
- Configurable thresholds
- Fast computation

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_timeseries.py
```

## Development

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Comprehensive docstrings
- Black formatting

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Performance

### Optimization Tips
- Use appropriate data types (datetime64, float32)
- Enable parallel processing where possible
- Cache model results
- Use efficient data structures

### Memory Management
- Process data in chunks for large datasets
- Clear unused variables
- Use generators for large sequences

## Troubleshooting

### Common Issues

1. **Prophet installation issues**
   ```bash
   pip install prophet --no-cache-dir
   ```

2. **PyTorch installation**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Memory issues with LSTM**
   - Reduce sequence length
   - Use smaller batch size
   - Enable gradient checkpointing

### Logging

Enable detailed logging by modifying `config/config.yaml`:

```yaml
logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Facebook Prophet team for the Prophet library
- Statsmodels contributors for ARIMA implementation
- PyTorch team for deep learning framework
- Streamlit team for the web interface framework


# Time-Series-Probabilistic-Forecasting
