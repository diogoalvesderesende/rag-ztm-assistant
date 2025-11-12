#!/usr/bin/env python3
"""
LangSmith tracing configuration for Time Series Course Assistant
"""

import os
from typing import Dict, Any

def get_langsmith_config() -> Dict[str, Any]:
    """Get LangSmith configuration from environment variables"""
    return {
        "tracing_enabled": os.getenv("LANGSMITH_TRACING", "true").lower() == "true",
        "endpoint": os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        "api_key": os.getenv("LANGSMITH_API_KEY"),
        "project": os.getenv("LANGSMITH_PROJECT", "time-series-course-assistant"),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "tags": ["time-series", "streamlit", "openai"],
    }

def is_langsmith_configured() -> bool:
    """Check if LangSmith is properly configured"""
    config = get_langsmith_config()
    return (
        config["tracing_enabled"] and 
        config["api_key"] is not None and 
        config["project"] is not None
    )

def get_environment_info() -> Dict[str, str]:
    """Get environment information for tracing"""
    return {
        "python_version": os.getenv("PYTHON_VERSION", "unknown"),
        "streamlit_version": os.getenv("STREAMLIT_VERSION", "unknown"),
        "deployment_platform": os.getenv("DEPLOYMENT_PLATFORM", "local"),
        "region": os.getenv("REGION", "unknown"),
        "langsmith_project": os.getenv("LANGSMITH_PROJECT", "unknown"),
    }
