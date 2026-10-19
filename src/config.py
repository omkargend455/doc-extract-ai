"""Global configuration and environment setup for DocExtract AI."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_RECEIPTS_DIR = DATA_DIR / "sample_receipts"
SAMPLE_INVOICES_DIR = DATA_DIR / "sample_invoices"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
REPORTS_DIR = BASE_DIR / "reports"
RESULTS_DIR = BASE_DIR / "results"

# Load local environment variables from .env
load_dotenv(BASE_DIR / ".env")

# API Keys & LLM settings (Optional for Milestone 1.1)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# System capability checks
TESSERACT_PATH = shutil.which("tesseract")
IS_TESSERACT_AVAILABLE = TESSERACT_PATH is not None


def is_gemini_available() -> bool:
    """Check if a Gemini API key is configured and available."""
    return bool(GEMINI_API_KEY and GEMINI_API_KEY.strip())


def get_system_status() -> dict[str, str | bool]:
    """Return runtime environment capability summary."""
    return {
        "base_dir": str(BASE_DIR),
        "gemini_configured": is_gemini_available(),
        "default_model": DEFAULT_GEMINI_MODEL,
        "tesseract_available": IS_TESSERACT_AVAILABLE,
        "tesseract_path": TESSERACT_PATH or "Not found in system PATH",
    }
