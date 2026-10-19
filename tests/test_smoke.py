"""Milestone 1.1 smoke tests for DocExtract AI environment and foundations."""

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
    from google import genai  # noqa: F401
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
        is_gemini_available,
        get_system_status,
        DEFAULT_GEMINI_MODEL,
    )

    assert BASE_DIR.exists()
    assert DATA_DIR.exists()
    assert isinstance(is_gemini_available(), bool)
    assert DEFAULT_GEMINI_MODEL != ""

    status = get_system_status()
    assert "base_dir" in status
    assert "gemini_configured" in status
    assert "tesseract_available" in status
