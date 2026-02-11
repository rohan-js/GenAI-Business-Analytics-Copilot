"""
Configuration module for the GenAI Business Analytics Copilot.
Centralizes all configuration constants and environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# Path Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# =============================================================================
# LLM Configuration
# =============================================================================

# Available models (ordered by recommendation)
AVAILABLE_MODELS = {
 "phi-2": {
 "name": "microsoft/phi-2",
 "description": "Fast, code-focused (2.7B params)",
 "context_length": 2048,
 },
 "gemma-2b": {
 "name": "google/gemma-2b-it",
 "description": "Balanced, instruction-tuned (2B params)",
 "context_length": 8192,
 },
 "tinyllama": {
 "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
 "description": "Ultra-fast, lightweight (1.1B params)",
 "context_length": 2048,
 },
}

# Default model
DEFAULT_MODEL = os.getenv("LLM_MODEL_NAME", "microsoft/phi-2")

# Model inference settings
MODEL_SETTINGS = {
 "max_new_tokens": 512,
 "temperature": 0.1, # Low for more deterministic code generation
 "top_p": 0.95,
 "do_sample": True,
 "use_cache": True,
}

# Quantization for memory efficiency
ENABLE_QUANTIZATION = os.getenv("ENABLE_MODEL_QUANTIZATION", "true").lower() == "true"

# =============================================================================
# Data Processing Configuration
# =============================================================================

# Maximum rows to process for LLM operations (large datasets are sampled)
MAX_SAMPLE_ROWS = int(os.getenv("MAX_SAMPLE_ROWS", "10000"))

# File size limits (in MB)
MAX_FILE_SIZE_MB = 100

# Supported file extensions
SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]

# Missing value indicators
MISSING_VALUE_INDICATORS = ["", "NA", "N/A", "null", "NULL", "None", "NaN", "-", "--"]

# =============================================================================
# Safety Configuration
# =============================================================================

# Sandbox settings
MAX_EXECUTION_SECONDS = int(os.getenv("MAX_EXECUTION_SECONDS", "30"))
MAX_MEMORY_MB = 512

# Allowed modules for code execution
ALLOWED_MODULES = frozenset({"pandas", "numpy", "pd", "np"})

# Blocked keywords in generated code
BLOCKED_KEYWORDS = frozenset({
 "exec", "eval", "compile", "open", "file",
 "import", "__import__", "importlib",
 "os", "sys", "subprocess", "shutil",
 "socket", "requests", "urllib",
 "pickle", "shelve", "marshal",
 "__builtins__", "__globals__", "__code__",
 "getattr", "setattr", "delattr",
})

# =============================================================================
# EDA Configuration
# =============================================================================

# Correlation threshold for "high correlation" flag
HIGH_CORRELATION_THRESHOLD = 0.7

# Outlier detection settings
OUTLIER_IQR_MULTIPLIER = 1.5
OUTLIER_ZSCORE_THRESHOLD = 3.0

# Maximum unique values for categorical treatment
MAX_CATEGORICAL_UNIQUE = 50

# =============================================================================
# Visualization Configuration
# =============================================================================

# Chart theme
CHART_THEME = "plotly_dark"

# Color palette for charts
COLOR_PALETTE = [
 "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
 "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
]

# Default chart dimensions
DEFAULT_CHART_WIDTH = 700
DEFAULT_CHART_HEIGHT = 450

# =============================================================================
# UI Configuration
# =============================================================================

# App metadata
APP_TITLE = " Business Analytics Copilot"
APP_SUBTITLE = "AI-Powered Insights for Business Decisions"

# Page configuration
PAGE_CONFIG = {
 "page_title": "Business Analytics Copilot",
 "page_icon": "",
 "layout": "wide",
 "initial_sidebar_state": "expanded",
}

# Theme
UI_THEME = os.getenv("THEME", "dark")
SHOW_DEBUG_INFO = os.getenv("SHOW_DEBUG_INFO", "false").lower() == "true"
