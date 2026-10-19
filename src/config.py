"""Global configuration and environment setup for DocExtract AI (NVIDIA NIM & Deterministic Baseline)."""

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

# NVIDIA NIM Hosted LLM Settings
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
DEFAULT_NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
SUPPORTED_NVIDIA_MODELS: list[str] = [
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]

# System capability checks
TESSERACT_PATH = shutil.which("tesseract")
IS_TESSERACT_AVAILABLE = TESSERACT_PATH is not None


def is_nvidia_available() -> bool:
    """Check if an NVIDIA API key is configured and available."""
    return bool(NVIDIA_API_KEY and NVIDIA_API_KEY.strip())


def get_system_status() -> dict[str, str | bool]:
    """Return runtime environment capability summary."""
    return {
        "base_dir": str(BASE_DIR),
        "nvidia_configured": is_nvidia_available(),
        "default_nvidia_model": DEFAULT_NVIDIA_MODEL,
        "tesseract_available": IS_TESSERACT_AVAILABLE,
        "tesseract_path": TESSERACT_PATH or "Not found in system PATH",
    }
