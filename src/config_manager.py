"""
Configuration management module for Time Series Analysis Project.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import os


class Config:
    """Configuration manager for the time series analysis project."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                self._config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self._config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = log_config.get('file', 'logs/timeseries.log')
        
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration dictionary
        """
        return self.get(f'models.{model_name}', {})
    
    def is_model_enabled(self, model_name: str) -> bool:
        """
        Check if a model is enabled.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model is enabled, False otherwise
        """
        return self.get(f'models.{model_name}.enable', False)
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get('data', {})
    
    def get_forecasting_config(self) -> Dict[str, Any]:
        """Get forecasting configuration."""
        return self.get('forecasting', {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self.get('visualization', {})


# Global configuration instance
config = Config()
