#!/usr/bin/env python3
"""
Demo script for Time Series Analysis Project.
This script demonstrates the basic functionality of the project.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config_manager import config
from data_utils import DataGenerator, save_data
from models import ProphetForecaster, ARIMAForecaster
from visualization import TimeSeriesVisualizer


def main():
    """Run a simple demo of the time series analysis project."""
    print("Time Series Analysis Project - Demo")
    print("=" * 50)
    
    # 1. Generate synthetic data
    print("\n1. Generating synthetic data...")
    generator = DataGenerator(config.get_data_config())
    data = generator.generate_synthetic_data()
    print(f"   Generated {len(data)} data points")
    print(f"   Date range: {data['ds'].min()} to {data['ds'].max()}")
    print(f"   Value range: {data['y'].min():.2f} to {data['y'].max():.2f}")
    
    # 2. Train Prophet model
    print("\n2. Training Prophet model...")
    prophet_config = config.get_model_config('prophet')
    prophet_model = ProphetForecaster(prophet_config)
    prophet_model.fit(data)
    print("   Prophet model trained successfully!")
    
    # 3. Generate forecast
    print("\n3. Generating forecast...")
    forecast_horizon = 30
    forecast = prophet_model.predict(forecast_horizon)
    confidence_intervals = prophet_model.get_confidence_intervals(forecast_horizon)
    print(f"   Generated {forecast_horizon}-day forecast")
    print(f"   Forecast range: {forecast['yhat'].min():.2f} to {forecast['yhat'].max():.2f}")
    
    # 4. Create visualization
    print("\n4. Creating visualization...")
    visualizer = TimeSeriesVisualizer(config.get_visualization_config())
    fig = visualizer.plot_forecast_with_intervals(
        data, 
        forecast, 
        confidence_intervals,
        "Prophet Forecast with Confidence Intervals"
    )
    
    # Save the plot
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    visualizer.save_plot(fig, output_dir / "demo_forecast.png")
    print(f"   Plot saved to {output_dir / 'demo_forecast.png'}")
    
    # 5. Save results
    print("\n5. Saving results...")
    save_data(data, output_dir / "demo_data.csv")
    save_data(forecast, output_dir / "demo_forecast.csv")
    print(f"   Data and forecast saved to {output_dir}/")
    
    # 6. Display summary
    print("\n6. Summary:")
    print(f"   - Data points: {len(data)}")
    print(f"   - Forecast horizon: {forecast_horizon} days")
    print(f"   - Confidence intervals: {list(confidence_intervals.keys())}")
    print(f"   - Output directory: {output_dir}")
    
    print("\nDemo completed successfully!")
    print("\nTo explore more features:")
    print("  - Run Streamlit dashboard: streamlit run app.py")
    print("  - Try command line interface: python main.py --mode cli")
    print("  - Open Jupyter notebook: jupyter notebook notebooks/")


if __name__ == "__main__":
    main()
