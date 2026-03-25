#!/usr/bin/env python3
"""
Configuration file for Rubber Ducky, the Course Assistant
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
PAGE_TITLE = "Rubber Ducky, the Course Assistant"
PAGE_ICON = "🦆"
LAYOUT = "wide"

# Chat Configuration
MAX_MESSAGES = 100  # Maximum messages to keep in history
CHAT_INPUT_PLACEHOLDER = "Ask Rubber Ducky anything about RAG, agents, or OpenAI apps!"

# Styling
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#9c27b0"
SUCCESS_COLOR = "#4caf50"
WARNING_COLOR = "#ff9800"
ERROR_COLOR = "#f44336"

# Example questions for the sidebar (keep them topic-focused, no course/lecture identifiers)
EXAMPLE_QUESTIONS = [
    "How does Rubber Ducky approach prompt engineering for RAG answers",
    "How do you design chunking and embeddings for better retrieval",
    "What is OpenAI File Search and how should I use it in a chat app",
    "How do I deploy a RAG Streamlit app to production",
    "How do unstructured documents flow through a retrieval + generation pipeline",
    "What are Whisper and CLIP used for in multimodal RAG",
    "How do RAGAS metrics help evaluate retrieval quality",
    "How does agentic RAG use state/memory to improve answers"
]
