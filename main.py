#!/usr/bin/env python3
"""
Main script for Time Series Analysis Project.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from config_manager import config
from data_utils import DataGenerator, DataPreprocessor, save_data, load_data
from models import ProphetForecaster, ARIMAForecaster, LSTMForecaster, AnomalyDetector, ModelEnsemble
from visualization import TimeSeriesVisualizer, MetricsVisualizer


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Time Series Analysis Project")
    parser.add_argument("--mode", choices=["cli", "streamlit"], default="streamlit",
                       help="Run mode: cli for command line, streamlit for web interface")
    parser.add_argument("--data-source", choices=["synthetic", "external"], default="synthetic",
                       help="Data source to use")
    parser.add_argument("--models", nargs="+", choices=["prophet", "arima", "lstm"], 
                       default=["prophet", "arima"], help="Models to train")
    parser.add_argument("--forecast-horizon", type=int, default=30,
                       help="Forecast horizon in days")
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Output directory for results")
    parser.add_argument("--anomaly-detection", action="store_true",
                       help="Enable anomaly detection")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = logging.getLogger(__name__)
    logger.info("Starting Time Series Analysis Project")
    
    if args.mode == "streamlit":
        # Launch Streamlit app
        import subprocess
        subprocess.run(["streamlit", "run", "app.py"])
    else:
        # Run CLI mode
        run_cli_mode(args)


def run_cli_mode(args):
    """Run the analysis in CLI mode."""
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate or load data
    if args.data_source == "synthetic":
        logger.info("Generating synthetic data...")
        generator = DataGenerator(config.get_data_config())
        data = generator.generate_synthetic_data()
        
        # Add anomalies if requested
        if args.anomaly_detection:
            data = generator.add_anomalies(data, anomaly_rate=0.05)
        
        # Save data
        save_data(data, output_dir / "synthetic_data.csv")
        logger.info(f"Generated {len(data)} data points")
    
    else:
        logger.info("Loading external data...")
        # This would load external data
        raise NotImplementedError("External data loading not implemented")
    
    # Initialize visualizer
    visualizer = TimeSeriesVisualizer(config.get_visualization_config())
    
    # Plot original data
    fig = visualizer.plot_time_series(data, "Original Time Series")
    visualizer.save_plot(fig, output_dir / "original_data.png")
    
    # Train models
    models = {}
    forecasts = {}
    confidence_intervals = {}
    
    for model_name in args.models:
        logger.info(f"Training {model_name} model...")
        
        try:
            if model_name == "prophet":
                model_config = config.get_model_config('prophet')
                model = ProphetForecaster(model_config)
                model.fit(data)
                models[model_name] = model
                
                # Generate forecast
                forecast = model.predict(args.forecast_horizon)
                forecasts[model_name] = forecast
                
                # Get confidence intervals
                intervals = model.get_confidence_intervals(args.forecast_horizon)
                confidence_intervals[model_name] = intervals
                
                logger.info(f"{model_name} model trained successfully")
            
            elif model_name == "arima":
                model_config = config.get_model_config('arima')
                model = ARIMAForecaster(model_config)
                model.fit(data)
                models[model_name] = model
                
                # Generate forecast
                forecast = model.predict(args.forecast_horizon)
                forecasts[model_name] = forecast
                
                # Get confidence intervals
                intervals = model.get_confidence_intervals(args.forecast_horizon)
                confidence_intervals[model_name] = intervals
                
                logger.info(f"{model_name} model trained successfully")
            
            elif model_name == "lstm":
                model_config = config.get_model_config('lstm')
                model = LSTMForecaster(model_config)
                model.fit(data)
                models[model_name] = model
                
                # Generate forecast
                forecast = model.predict(args.forecast_horizon)
                forecasts[model_name] = forecast
                
                # Get confidence intervals
                intervals = model.get_confidence_intervals(args.forecast_horizon)
                confidence_intervals[model_name] = intervals
                
                logger.info(f"{model_name} model trained successfully")
        
        except Exception as e:
            logger.error(f"Failed to train {model_name} model: {e}")
    
    # Plot forecasts
    if forecasts:
        fig = visualizer.plot_forecast_with_intervals(
            data, 
            list(forecasts.values())[0],  # Use first forecast for intervals
            confidence_intervals.get(list(forecasts.keys())[0]),
            "Forecast with Confidence Intervals"
        )
        visualizer.save_plot(fig, output_dir / "forecast.png")
        
        # Plot model comparison
        fig = visualizer.plot_model_comparison(forecasts, data, "Model Comparison")
        visualizer.save_plot(fig, output_dir / "model_comparison.png")
    
    # Anomaly detection
    if args.anomaly_detection:
        logger.info("Performing anomaly detection...")
        
        detector = AnomalyDetector(config.get_model_config('anomaly_detection'))
        
        # Isolation Forest
        data_with_anomalies = detector.detect_with_isolation_forest(data)
        
        # Statistical method
        data_with_anomalies = detector.detect_with_statistical_method(data_with_anomalies)
        
        # Plot anomalies
        fig = visualizer.plot_anomalies(data_with_anomalies, "is_anomaly_if", "Anomaly Detection Results")
        visualizer.save_plot(fig, output_dir / "anomalies.png")
        
        # Save anomaly results
        save_data(data_with_anomalies, output_dir / "data_with_anomalies.csv")
        
        logger.info("Anomaly detection completed")
    
    # Save forecasts
    for model_name, forecast in forecasts.items():
        save_data(forecast, output_dir / f"{model_name}_forecast.csv")
    
    logger.info(f"Analysis completed. Results saved to {output_dir}")
    print(f"Analysis completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
