"""DocExtract AI — Streamlit Smoke Test Application (Milestone 1.1).

This minimal application verifies environment configuration, foundational
dependencies, and runtime UI responsiveness before pipeline implementation.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import get_system_status, is_gemini_available, DEFAULT_GEMINI_MODEL

# Configure Streamlit page
st.set_page_config(
    page_title="DocExtract AI — Foundation Smoke Test",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📄 DocExtract AI")
st.caption(
    "Schema-Driven Document Extraction & Deterministic Validation Pipeline | Milestone 1.1 Foundation"
)

st.divider()

# System Status Overview
st.subheader("System Status & Environment Verification")
status = get_system_status()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Python Runtime", value=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

with col2:
    gemini_ready = is_gemini_available()
    st.metric(
        label="Gemini API Key",
        value="Configured" if gemini_ready else "Optional (Unset)",
        delta="Ready" if gemini_ready else "Offline Mode",
        delta_color="normal" if gemini_ready else "off",
    )

with col3:
    tesseract_ready = status["tesseract_available"]
    st.metric(
        label="Tesseract OCR",
        value="Available" if tesseract_ready else "Not on PATH",
        delta="Native" if tesseract_ready else "Cloud Fallback",
        delta_color="normal" if tesseract_ready else "off",
    )

with col4:
    st.metric(label="Default Gemini Model", value=DEFAULT_GEMINI_MODEL)

st.divider()

# Tab Skeleton Preview
tab_extract, tab_validate, tab_benchmark = st.tabs(
    ["Extraction & Inspection", "Validation & Review Queue", "Benchmark Analytics"]
)

with tab_extract:
    st.markdown("### Document Ingestion & Extraction")
    st.info(
        "**Milestone 1.1 Checkpoint Passed:** Streamlit engine is active and responding. "
        "Extraction pipelines (Track A Regex & Track B Gemini 1.5 Flash) will be enabled in subsequent milestones."
    )
    uploaded_file = st.file_uploader(
        "Smoke Test Upload (PDF, PNG, JPG)",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Upload verification widget to confirm Streamlit file-handling reactivity.",
    )
    if uploaded_file is not None:
        st.success(f"File uploaded successfully: `{uploaded_file.name}` ({uploaded_file.size} bytes)")

with tab_validate:
    st.markdown("### Deterministic Business Rules & Review Decider")
    st.write(
        "This tab will host arithmetic balance verification, date chronological checks, "
        "and human-in-the-loop audit routing in Milestone 1.4."
    )

with tab_benchmark:
    st.markdown("### Empirical Performance Benchmark")
    st.write(
        "This tab will display live and pre-computed evaluation matrices (Exact Match, RapidFuzz F1, "
        "Latency, and API Cost) across the SROIE benchmark dataset in Milestone 2.1."
    )

# Sidebar Information
with st.sidebar:
    st.header("Project Details")
    st.markdown(
        """
        - **Project:** DocExtract AI (`cynbit-doc-extraction`)
        - **Target:** Cynbit AI/ML Intern Assessment
        - **Milestone:** 1.1 Foundation & Smoke Test
        - **Status:** All core dependencies verified
        """
    )
    st.markdown("---")
    st.markdown("### Environment Details")
    st.json(status)
