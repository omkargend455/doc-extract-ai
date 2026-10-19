"""Smoke tests for DocExtract AI environment and foundations (NVIDIA NIM & Deterministic Baseline)."""

import sys
from pathlib import Path

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def test_core_dependencies_import():
    """Verify that all pinned foundational libraries import successfully."""
    import streamlit  # noqa: F401
    import pydantic  # noqa: F401
    import openai  # noqa: F401
    import fitz  # noqa: F401 (PyMuPDF)
    from PIL import Image  # noqa: F401
    import pytesseract  # noqa: F401
    import rapidfuzz  # noqa: F401
    import pandas  # noqa: F401
    import plotly  # noqa: F401
    import dotenv  # noqa: F401

    assert True


def test_config_paths_and_fallbacks():
    """Verify path definitions and system capability detection."""
    from src.config import (
        BASE_DIR,
        DATA_DIR,
        is_nvidia_available,
        get_system_status,
        DEFAULT_NVIDIA_MODEL,
        SUPPORTED_NVIDIA_MODELS,
    )

    assert BASE_DIR.exists()
    assert DATA_DIR.exists()
    assert isinstance(is_nvidia_available(), bool)
    assert DEFAULT_NVIDIA_MODEL != ""
    assert "nvidia/nemotron-3.5-lightning-30b-a3b" in SUPPORTED_NVIDIA_MODELS

    status = get_system_status()
    assert "base_dir" in status
    assert "nvidia_configured" in status
    assert "tesseract_available" in status
