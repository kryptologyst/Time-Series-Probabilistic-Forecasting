#!/usr/bin/env python3
"""
Setup script for Time Series Analysis Project.
"""

import subprocess
import sys
from pathlib import Path


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True


def create_directories():
    """Create necessary directories."""
    directories = ["data", "models", "logs", "output"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")


def run_tests():
    """Run unit tests."""
    print("Running unit tests...")
    try:
        subprocess.check_call([sys.executable, "-m", "pytest", "tests/", "-v"])
        print("All tests passed!")
    except subprocess.CalledProcessError as e:
        print(f"Some tests failed: {e}")
        return False
    return True


def main():
    """Main setup function."""
    print("Setting up Time Series Analysis Project...")
    
    # Install requirements
    if not install_requirements():
        print("Setup failed during package installation")
        return
    
    # Create directories
    create_directories()
    
    # Run tests
    if not run_tests():
        print("Setup completed with test failures")
        return
    
    print("\nSetup completed successfully!")
    print("\nTo run the project:")
    print("  Streamlit dashboard: streamlit run app.py")
    print("  Command line: python main.py --mode cli")
    print("  Jupyter notebook: jupyter notebook notebooks/")


if __name__ == "__main__":
    main()
