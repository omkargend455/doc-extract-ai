"""DocExtract AI — Streamlit Application (Milestone 1.2: Ingestion & Ground Truth).

Demonstrates digital PDF ingestion via PyMuPDF, image OCR capability checking
with actionable guidance, and inspection of synthetic documents with paired ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    get_system_status,
    is_gemini_available,
    DEFAULT_GEMINI_MODEL,
    SAMPLE_INVOICES_DIR,
    SAMPLE_RECEIPTS_DIR,
    GROUND_TRUTH_DIR,
)
from src.ingest import ingest_document, IngestionResult, TESSERACT_GUIDANCE_MESSAGE
from src.schemas import load_ground_truth

# Configure Streamlit page
st.set_page_config(
    page_title="DocExtract AI — Ingestion & Ground Truth",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📄 DocExtract AI")
st.caption(
    "Schema-Driven Document Extraction & Deterministic Validation Pipeline | Milestone 1.2: Ingestion & Ground Truth"
)

st.divider()

# System Status Overview
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
        delta="Native" if tesseract_ready else "Actionable Fallback",
        delta_color="normal" if tesseract_ready else "off",
    )

with col4:
    st.metric(label="Default Gemini Model", value=DEFAULT_GEMINI_MODEL)

st.divider()

# 3-Tab Structure
tab_extract, tab_validate, tab_benchmark = st.tabs(
    ["1. Ingestion & Document Inspector", "2. Validation & Review Queue", "3. Benchmark Analytics"]
)

with tab_extract:
    st.markdown("### Document Ingestion & Text Layer Preview")
    st.info(
        "**Milestone 1.2 Active:** Digital PDF ingestion via PyMuPDF and Tesseract image routing are enabled. "
        "Extraction pipelines (Track A Regex & Track B Gemini) will be hooked in Milestone 1.3/1.4."
    )

    doc_mode = st.radio(
        "Select Document Source:",
        ["Pre-Generated Synthetic Documents", "Upload Custom Document (PDF / Image)"],
        horizontal=True,
    )

    selected_doc_path: Path | None = None
    uploaded_file = None
    ground_truth_obj = None

    if doc_mode == "Pre-Generated Synthetic Documents":
        all_samples = sorted(list(SAMPLE_INVOICES_DIR.glob("*.pdf")) + list(SAMPLE_RECEIPTS_DIR.glob("*.pdf")))
        sample_names = [f.name for f in all_samples]

        if sample_names:
            chosen_name = st.selectbox("Select a sample document:", sample_names)
            selected_doc_path = next(p for p in all_samples if p.name == chosen_name)

            # Check for paired ground truth
            gt_path = GROUND_TRUTH_DIR / f"{selected_doc_path.stem}.json"
            if gt_path.exists():
                ground_truth_obj = load_ground_truth(gt_path)
        else:
            st.warning("No synthetic documents found. Run `python data/generate_synthetic.py` to create the testbed.")

    else:
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["pdf", "png", "jpg", "jpeg", "txt"],
            help="Upload a digital PDF invoice or scanned image receipt.",
        )

    # Process Document Ingestion
    doc_target = uploaded_file or selected_doc_path

    if doc_target is not None:
        filename = getattr(doc_target, "name", str(selected_doc_path.name if selected_doc_path else "doc"))
        ingestion_res = ingest_document(doc_target, filename=filename)

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown(f"#### Ingested Document: `{filename}`")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Source Type", ingestion_res.source_type)
            m_col2.metric("Page Count", ingestion_res.page_count)
            m_col3.metric("Characters", ingestion_res.character_count)

            if ingestion_res.warning:
                st.warning(ingestion_res.warning)

            st.markdown("##### Extracted Text Layer")
            if ingestion_res.text:
                st.text_area("Extracted Raw Text", value=ingestion_res.text, height=350)
            else:
                st.info("No raw text was extracted from this file.")

        with col_right:
            st.markdown("#### Paired Ground Truth Annotation")
            if ground_truth_obj:
                st.success(f"Ground Truth Loaded for `{ground_truth_obj.doc_id}` ({ground_truth_obj.doc_type})")
                st.json(ground_truth_obj.model_dump(exclude={"raw_text"}))
                if getattr(ground_truth_obj, "injected_errors", None):
                    st.warning(f"Injected Test Edge Cases: {ground_truth_obj.injected_errors}")
            else:
                st.info(
                    "No ground-truth annotation exists for this file. "
                    "Pre-generated synthetic files include paired ground truth in `data/ground_truth/`."
                )

with tab_validate:
    st.markdown("### Deterministic Business Rules & Review Decider")
    st.info(
        "Validation rules ($\text{Subtotal} + \text{Tax} - \text{Discount} == \text{Total}$, "
        "chronological checks, and the Composite Review Decider) will be implemented in Milestone 1.4."
    )

with tab_benchmark:
    st.markdown("### Empirical Performance Benchmark")
    st.info(
        "Evaluation metrics (Exact Match, RapidFuzz $F_1$, Latency, and Cost per 1,000 docs) "
        "will be implemented in Milestone 2.1."
    )

# Sidebar Information
with st.sidebar:
    st.header("Project Details")
    st.markdown(
        """
        - **Project:** DocExtract AI (`cynbit-doc-extraction`)
        - **Target:** Cynbit AI/ML Intern Assessment
        - **Milestone:** 1.2 Data Ingestion & Ground Truth
        - **Status:** Ingestion & Synthetic Testbed Verified
        """
    )
    st.markdown("---")
    st.markdown("### Ingestion Capabilities")
    st.markdown(
        f"""
        - **Digital PDFs:** Native via PyMuPDF
        - **Scanned Images:** Tesseract OCR ({'Installed' if tesseract_ready else 'Not on PATH'})
        - **Gemini API:** {'Configured' if gemini_ready else 'Offline Mode'}
        """
    )
    if not tesseract_ready:
        with st.expander("Tesseract Setup Guide"):
            st.caption(TESSERACT_GUIDANCE_MESSAGE)
