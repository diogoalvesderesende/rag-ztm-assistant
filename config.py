#!/usr/bin/env python3
"""
Configuration file for Time Series Course Assistant
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Vector store configuration
# Vector store ID is loaded from Streamlit secrets
VECTOR_STORE_ID = None  # Will be loaded from secrets

# API Configuration
DEFAULT_MODEL = "gpt-5-nano"
AVAILABLE_MODELS = [
    "gpt-5-nano",
    "gpt-4", 
    "claude-3-sonnet"
]

# Verbosity levels
VERBOSITY_LEVELS = ["low", "medium", "high"]

# UI Configuration
PAGE_TITLE = "Time Series Course Assistant"
PAGE_ICON = "🚀"
LAYOUT = "wide"

# Chat Configuration
MAX_MESSAGES = 100  # Maximum messages to keep in history
CHAT_INPUT_PLACEHOLDER = "Ask me anything about Time Series!"

# Styling
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#9c27b0"
SUCCESS_COLOR = "#4caf50"
WARNING_COLOR = "#ff9800"
ERROR_COLOR = "#f44336"

# Example questions for the sidebar
EXAMPLE_QUESTIONS = [
    "ARIMA vs SARIMA—when to use each?",
    "Show the LSTM code we used",
    "When to use NBEATS vs TFT?",
    "How do we implement cross-validation?",
    "What are the pros/cons of Prophet?",
    "How do we handle seasonality in time series?",
    "What's the difference between MAE and RMSE?",
    "How do we add exogenous variables to SARIMAX?"
]
