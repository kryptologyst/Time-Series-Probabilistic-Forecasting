"""
Streamlit dashboard for Time Series Analysis Project.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config_manager import config
from data_utils import DataGenerator, DataPreprocessor, save_data, load_data
from models import ProphetForecaster, ARIMAForecaster, LSTMForecaster, AnomalyDetector, ModelEnsemble
from visualization import TimeSeriesVisualizer, MetricsVisualizer

# Page configuration
st.set_page_config(
    page_title="Time Series Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Time Series Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Synthetic Data", "Upload CSV", "External Dataset"]
    )
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = {}
    if 'anomalies' not in st.session_state:
        st.session_state.anomalies = None
    
    # Data loading section
    with st.expander("📊 Data Loading", expanded=True):
        if data_source == "Synthetic Data":
            st.subheader("Generate Synthetic Data")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                periods = st.number_input("Number of Periods", min_value=100, max_value=5000, value=1000)
                trend_start = st.number_input("Trend Start", min_value=0.0, max_value=100.0, value=10.0)
            with col2:
                trend_end = st.number_input("Trend End", min_value=0.0, max_value=100.0, value=50.0)
                seasonal_amplitude = st.number_input("Seasonal Amplitude", min_value=0.0, max_value=20.0, value=5.0)
            with col3:
                noise_std = st.number_input("Noise Standard Deviation", min_value=0.0, max_value=10.0, value=2.0)
                add_anomalies = st.checkbox("Add Anomalies", value=True)
            
            if st.button("Generate Data", type="primary"):
                with st.spinner("Generating synthetic data..."):
                    # Update config
                    config._config['data']['synthetic']['periods'] = periods
                    config._config['data']['synthetic']['trend_start'] = trend_start
                    config._config['data']['synthetic']['trend_end'] = trend_end
                    config._config['data']['synthetic']['seasonal_amplitude'] = seasonal_amplitude
                    config._config['data']['synthetic']['noise_std'] = noise_std
                    
                    # Generate data
                    generator = DataGenerator(config.get_data_config())
                    data = generator.generate_synthetic_data()
                    
                    if add_anomalies:
                        data = generator.add_anomalies(data, anomaly_rate=0.05)
                    
                    st.session_state.data = data
                    st.success(f"Generated {len(data)} data points!")
        
        elif data_source == "Upload CSV":
            st.subheader("Upload CSV File")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    
                    # Display column selection
                    col1, col2 = st.columns(2)
                    with col1:
                        date_col = st.selectbox("Select Date Column", data.columns)
                    with col2:
                        value_col = st.selectbox("Select Value Column", data.columns)
                    
                    if st.button("Load Data", type="primary"):
                        # Convert to standard format
                        data['ds'] = pd.to_datetime(data[date_col])
                        data['y'] = data[value_col]
                        data = data[['ds', 'y']].dropna()
                        
                        st.session_state.data = data
                        st.success(f"Loaded {len(data)} data points!")
                
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        
        elif data_source == "External Dataset":
            st.subheader("External Dataset")
            dataset_choice = st.selectbox("Select Dataset", ["Energy Consumption", "Traffic Data", "Weather Data"])
            
            if st.button("Load External Data", type="primary"):
                with st.spinner("Loading external data..."):
                    # This would load from external sources
                    st.info("External data loading not implemented in this demo. Please use synthetic data or upload your own CSV.")
    
    # Display loaded data
    if st.session_state.data is not None:
        st.subheader("📈 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Points", len(st.session_state.data))
        with col2:
            st.metric("Date Range", f"{(st.session_state.data['ds'].max() - st.session_state.data['ds'].min()).days} days")
        with col3:
            st.metric("Mean Value", f"{st.session_state.data['y'].mean():.2f}")
        with col4:
            st.metric("Std Deviation", f"{st.session_state.data['y'].std():.2f}")
        
        # Plot data
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.data['ds'],
            y=st.session_state.data['y'],
            mode='lines',
            name='Time Series',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add anomaly markers if present
        if 'is_anomaly' in st.session_state.data.columns:
            anomaly_data = st.session_state.data[st.session_state.data['is_anomaly']]
            if not anomaly_data.empty:
                fig.add_trace(go.Scatter(
                    x=anomaly_data['ds'],
                    y=anomaly_data['y'],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=8)
                ))
        
        fig.update_layout(
            title="Time Series Data",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data statistics
        st.subheader("📊 Data Statistics")
        st.dataframe(st.session_state.data.describe())
    
    # Model training section
    if st.session_state.data is not None:
        st.subheader("🤖 Model Training")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            enable_prophet = st.checkbox("Prophet", value=True)
        with col2:
            enable_arima = st.checkbox("ARIMA", value=True)
        with col3:
            enable_lstm = st.checkbox("LSTM", value=False)
        
        forecast_horizon = st.slider("Forecast Horizon (days)", min_value=7, max_value=365, value=30)
        
        if st.button("Train Models", type="primary"):
            with st.spinner("Training models..."):
                models = {}
                
                if enable_prophet:
                    try:
                        prophet_config = config.get_model_config('prophet')
                        prophet_model = ProphetForecaster(prophet_config)
                        prophet_model.fit(st.session_state.data)
                        models['Prophet'] = prophet_model
                        st.success("Prophet model trained successfully!")
                    except Exception as e:
                        st.error(f"Prophet training failed: {e}")
                
                if enable_arima:
                    try:
                        arima_config = config.get_model_config('arima')
                        arima_model = ARIMAForecaster(arima_config)
                        arima_model.fit(st.session_state.data)
                        models['ARIMA'] = arima_model
                        st.success("ARIMA model trained successfully!")
                    except Exception as e:
                        st.error(f"ARIMA training failed: {e}")
                
                if enable_lstm:
                    try:
                        lstm_config = config.get_model_config('lstm')
                        lstm_model = LSTMForecaster(lstm_config)
                        lstm_model.fit(st.session_state.data)
                        models['LSTM'] = lstm_model
                        st.success("LSTM model trained successfully!")
                    except Exception as e:
                        st.error(f"LSTM training failed: {e}")
                
                st.session_state.models = models
        
        # Forecasting section
        if st.session_state.models:
            st.subheader("🔮 Forecasting")
            
            if st.button("Generate Forecasts", type="primary"):
                with st.spinner("Generating forecasts..."):
                    forecasts = {}
                    confidence_intervals = {}
                    
                    for model_name, model in st.session_state.models.items():
                        try:
                            forecast = model.predict(forecast_horizon)
                            forecasts[model_name] = forecast
                            
                            # Get confidence intervals
                            intervals = model.get_confidence_intervals(forecast_horizon)
                            confidence_intervals[model_name] = intervals
                            
                        except Exception as e:
                            st.error(f"Forecast generation failed for {model_name}: {e}")
                    
                    st.session_state.forecasts = forecasts
                    st.session_state.confidence_intervals = confidence_intervals
                    st.success("Forecasts generated successfully!")
            
            # Display forecasts
            if st.session_state.forecasts:
                st.subheader("📈 Forecast Results")
                
                # Create forecast comparison plot
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=st.session_state.data['ds'],
                    y=st.session_state.data['y'],
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='black', width=2)
                ))
                
                # Add forecasts
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                for i, (model_name, forecast) in enumerate(st.session_state.forecasts.items()):
                    color = colors[i % len(colors)]
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat'],
                        mode='lines',
                        name=f'{model_name} Forecast',
                        line=dict(color=color, width=2, dash='dash')
                    ))
                    
                    # Add confidence intervals if available
                    if model_name in st.session_state.confidence_intervals:
                        intervals = st.session_state.confidence_intervals[model_name]
                        if '95%' in intervals:
                            ci_df = intervals['95%']
                            fig.add_trace(go.Scatter(
                                x=ci_df['ds'],
                                y=ci_df['yhat_upper'],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                            fig.add_trace(go.Scatter(
                                x=ci_df['ds'],
                                y=ci_df['yhat_lower'],
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                                name=f'{model_name} 95% CI',
                                hoverinfo='skip'
                            ))
                
                fig.update_layout(
                    title="Forecast Comparison",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    hovermode='x unified',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast table
                st.subheader("📋 Forecast Values")
                forecast_tabs = st.tabs(list(st.session_state.forecasts.keys()))
                
                for i, (model_name, forecast) in enumerate(st.session_state.forecasts.items()):
                    with forecast_tabs[i]:
                        st.dataframe(forecast.head(10))
                        
                        # Download button
                        csv = forecast.to_csv(index=False)
                        st.download_button(
                            label=f"Download {model_name} Forecast",
                            data=csv,
                            file_name=f"{model_name.lower()}_forecast.csv",
                            mime="text/csv"
                        )
        
        # Anomaly detection section
        st.subheader("🚨 Anomaly Detection")
        
        col1, col2 = st.columns(2)
        with col1:
            enable_isolation_forest = st.checkbox("Isolation Forest", value=True)
        with col2:
            enable_statistical = st.checkbox("Statistical Method", value=True)
        
        if st.button("Detect Anomalies", type="primary"):
            with st.spinner("Detecting anomalies..."):
                anomaly_config = config.get_model_config('anomaly_detection')
                detector = AnomalyDetector(anomaly_config)
                
                data_with_anomalies = st.session_state.data.copy()
                
                if enable_isolation_forest:
                    try:
                        data_with_anomalies = detector.detect_with_isolation_forest(data_with_anomalies)
                        st.success("Isolation Forest anomaly detection completed!")
                    except Exception as e:
                        st.error(f"Isolation Forest failed: {e}")
                
                if enable_statistical:
                    try:
                        data_with_anomalies = detector.detect_with_statistical_method(data_with_anomalies)
                        st.success("Statistical anomaly detection completed!")
                    except Exception as e:
                        st.error(f"Statistical method failed: {e}")
                
                st.session_state.anomalies = data_with_anomalies
        
        # Display anomaly results
        if st.session_state.anomalies is not None:
            st.subheader("🔍 Anomaly Detection Results")
            
            # Anomaly summary
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'is_anomaly_if' in st.session_state.anomalies.columns:
                    if_count = st.session_state.anomalies['is_anomaly_if'].sum()
                    st.metric("Isolation Forest Anomalies", if_count)
            with col2:
                if 'is_anomaly_stat' in st.session_state.anomalies.columns:
                    stat_count = st.session_state.anomalies['is_anomaly_stat'].sum()
                    st.metric("Statistical Anomalies", stat_count)
            with col3:
                total_anomalies = len(st.session_state.anomalies)
                st.metric("Total Data Points", total_anomalies)
            
            # Anomaly plot
            fig = go.Figure()
            
            # Normal data
            normal_data = st.session_state.anomalies[
                ~(st.session_state.anomalies.get('is_anomaly_if', False) | 
                  st.session_state.anomalies.get('is_anomaly_stat', False))
            ]
            fig.add_trace(go.Scatter(
                x=normal_data['ds'],
                y=normal_data['y'],
                mode='lines',
                name='Normal Data',
                line=dict(color='#1f77b4', width=1)
            ))
            
            # Anomalies
            if 'is_anomaly_if' in st.session_state.anomalies.columns:
                if_anomalies = st.session_state.anomalies[st.session_state.anomalies['is_anomaly_if']]
                if not if_anomalies.empty:
                    fig.add_trace(go.Scatter(
                        x=if_anomalies['ds'],
                        y=if_anomalies['y'],
                        mode='markers',
                        name='Isolation Forest Anomalies',
                        marker=dict(color='red', size=8)
                    ))
            
            if 'is_anomaly_stat' in st.session_state.anomalies.columns:
                stat_anomalies = st.session_state.anomalies[st.session_state.anomalies['is_anomaly_stat']]
                if not stat_anomalies.empty:
                    fig.add_trace(go.Scatter(
                        x=stat_anomalies['ds'],
                        y=stat_anomalies['y'],
                        mode='markers',
                        name='Statistical Anomalies',
                        marker=dict(color='orange', size=8)
                    ))
            
            fig.update_layout(
                title="Anomaly Detection Results",
                xaxis_title="Date",
                yaxis_title="Value",
                hovermode='x unified',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Time Series Analysis Dashboard | Built with Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
