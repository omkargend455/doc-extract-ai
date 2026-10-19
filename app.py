"""DocExtract AI — Streamlit Application (Milestones 2.1 & 2.2).

An enterprise-ready, schema-driven document extraction, deterministic validation,
and confidence-routing pipeline. Supports digital PDFs, OCR image receipts,
offline rule-based baseline fallback, and live benchmark analytics.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    get_system_status,
    is_nvidia_available,
    DEFAULT_NVIDIA_MODEL,
    NVIDIA_BASE_URL,
    SAMPLE_INVOICES_DIR,
    SAMPLE_RECEIPTS_DIR,
    GROUND_TRUTH_DIR,
    RESULTS_DIR,
)
from src.ingest import ingest_document, IngestionResult, TESSERACT_GUIDANCE_MESSAGE
from src.schemas import load_ground_truth, ReviewStatus, ExtractionResult, DocumentValidationSummary, ReviewDecision
from src.baseline_regex import RegexBaselineExtractor
from src.nvidia_extractor import NvidiaExtractor, get_configured_nvidia_api_key
from src.extractor_dispatcher import extract_with_provider
from src.validators import validate_document
from src.confidence import determine_review_routing
from src.evaluate import match_field

# -----------------------------------------------------------------------------
# STREAMLIT CONFIGURATION & CUSTOM STYLES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DocExtract AI — Document Extraction & Validation",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e9ecef;
    }
    .status-badge-approved {
        background-color: #d4edda;
        color: #155724;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
    }
    .status-badge-warning {
        background-color: #fff3cd;
        color: #856404;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
    }
    .status-badge-review {
        background-color: #f8d7da;
        color: #721c24;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "last_extracted_id" not in st.session_state:
    st.session_state["last_extracted_id"] = None
if "extraction_result" not in st.session_state:
    st.session_state["extraction_result"] = None
if "validation_summary" not in st.session_state:
    st.session_state["validation_summary"] = None
if "review_decision" not in st.session_state:
    st.session_state["review_decision"] = None
if "manual_approved" not in st.session_state:
    st.session_state["manual_approved"] = False

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & CAPABILITY METRICS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Engine Settings")

    # Provider Mode Selector
    engine_choice = st.selectbox(
        "Extraction Provider:",
        [
            "Auto (NVIDIA NIM -> Offline Baseline)",
            "NVIDIA NIM (Hosted LLM)",
            "Offline Regex Baseline (Zero API calls)",
        ],
        index=0,
        help="Select extraction engine. Auto prioritizes hosted NVIDIA NIM if an API key is configured, otherwise runs the deterministic offline baseline.",
    )

    st.markdown("#### NVIDIA NIM Credentials & Models")

    # NVIDIA Settings
    env_nvidia_key = get_configured_nvidia_api_key()
    user_nvidia_key = st.text_input(
        "NVIDIA API Key (Optional):",
        value=env_nvidia_key or "",
        type="password",
        help="Enter your NVIDIA API key (nvapi-...) or set NVIDIA_API_KEY in your .env file.",
    )
    active_nvidia_key = user_nvidia_key.strip() if user_nvidia_key.strip() else None

    selected_nvidia_model = st.selectbox(
        "NVIDIA Model:",
        [
            "nvidia/nemotron-3.5-lightning-30b-a3b",
            "nvidia/nemotron-3.5-super-vl-120b-a12b",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
        ],
        index=0,
        help="Select NVIDIA NIM model. Nemotron 3.5 Lightning is optimized for fast OCR-text extraction; Super VL 120B supports multimodal vision+text; GPT-OSS models serve as hosted fallback.",
    )

    st.divider()

    # System Status Overview
    sys_status = get_system_status()
    st.markdown("### System Diagnostics")
    st.markdown(
        f"""
        - **Python:** `{sys.version.split()[0]}`
        - **NVIDIA Status:** `{'Ready (Key Configured)' if active_nvidia_key else 'Unconfigured'}`
        - **Tesseract OCR:** `{'Available' if sys_status['tesseract_available'] else 'Not on PATH'}`
        """
    )

    if not sys_status["tesseract_available"]:
        with st.expander("Tesseract Setup Guide"):
            st.caption(TESSERACT_GUIDANCE_MESSAGE)

    st.divider()
    st.caption("Cynbit AI/ML Technical Assessment | DocExtract AI v2.3")

# -----------------------------------------------------------------------------
# HEADER & SYSTEM METRICS
# -----------------------------------------------------------------------------
st.title("📄 DocExtract AI")
st.caption(
    "Schema-Driven Document Information Extraction, Deterministic Business Rule Validation, "
    "and Human-in-the-Loop Review Routing"
)

# Determine active pipeline label
if "offline" in engine_choice.lower() or "baseline" in engine_choice.lower():
    active_pipe_label = "Offline Baseline"
elif "nvidia" in engine_choice.lower() and "auto" not in engine_choice.lower():
    short_model_name = selected_nvidia_model.split("/")[-1]
    active_pipe_label = f"NVIDIA ({short_model_name})"
else:
    if active_nvidia_key:
        active_pipe_label = f"Auto (NVIDIA: {selected_nvidia_model})"
    else:
        active_pipe_label = "Auto (Offline Baseline)"

top_c1, top_c2, top_c3, top_c4 = st.columns(4)
with top_c1:
    st.metric(label="Active Pipeline", value=active_pipe_label)
with top_c2:
    st.metric(label="Validation Engine", value="Deterministic Rules", delta="Tolerance ±0.05")
with top_c3:
    st.metric(label="Routing Decider", value="3-Tier Composite", delta="Evidence-Based")
with top_c4:
    st.metric(label="Target Documents", value="Invoices & Receipts", delta="A4 PDF + Thermal")

st.divider()

# -----------------------------------------------------------------------------
# MAIN 3-TAB INTERFACE
# -----------------------------------------------------------------------------
tab_extract, tab_validate, tab_benchmark = st.tabs(
    ["1. Ingestion & Document Extraction", "2. Validation & Review Queue", "3. Benchmark Analytics"]
)

# =============================================================================
# TAB 1: INGESTION & DOCUMENT EXTRACTION
# =============================================================================
with tab_extract:
    st.markdown("### 📥 Document Ingestion & Extraction")

    col_input_left, col_input_right = st.columns([1, 1])

    with col_input_left:
        doc_source = st.radio(
            "Select Document Source:",
            ["Pre-Generated Synthetic Testbed", "Upload Custom Document (PDF / Image / TXT)"],
            horizontal=True,
        )

        selected_doc_path: Path | None = None
        uploaded_file = None
        ground_truth_obj = None

        if doc_source == "Pre-Generated Synthetic Testbed":
            all_samples = sorted(list(SAMPLE_INVOICES_DIR.glob("*.pdf")) + list(SAMPLE_RECEIPTS_DIR.glob("*.pdf")))
            sample_names = [f.name for f in all_samples]

            if sample_names:
                chosen_sample = st.selectbox(
                    "Choose a testbed document (includes intentional edge cases):",
                    sample_names,
                    index=0,
                )
                selected_doc_path = next(p for p in all_samples if p.name == chosen_sample)
                gt_path = GROUND_TRUTH_DIR / f"{selected_doc_path.stem}.json"
                if gt_path.exists():
                    ground_truth_obj = load_ground_truth(gt_path)
            else:
                st.warning("No synthetic testbed documents found. Run `python data/generate_synthetic.py`.")
        else:
            uploaded_file = st.file_uploader(
                "Upload invoice or receipt:",
                type=["pdf", "png", "jpg", "jpeg", "txt"],
                help="Digital PDFs are parsed natively via PyMuPDF. Scanned images use OCR.",
            )

    doc_target = uploaded_file or selected_doc_path

    if doc_target is not None:
        filename = getattr(doc_target, "name", str(selected_doc_path.name if selected_doc_path else "doc"))
        ingestion_res = ingest_document(doc_target, filename=filename)

        with col_input_right:
            st.markdown(f"#### Ingestion Preview: `{filename}`")

            m1, m2, m3 = st.columns(3)
            m1.metric("Format", ingestion_res.source_type)
            m2.metric("Pages", ingestion_res.page_count)
            m3.metric("Characters", ingestion_res.character_count)

            if ingestion_res.warning:
                st.warning(ingestion_res.warning)

            # Auto-detect document schema from content (tax invoice keywords vs receipt)
            text_lower = (ingestion_res.text or "").lower()
            invoice_markers = ["tax invoice", "invoice number", "gstin", "sold by", "bill to", "due date", "subtotal", "cgst", "sgst", "igst"]
            is_likely_invoice = any(kw in text_lower for kw in invoice_markers) or ("inv" in filename.lower())
            default_index = 0 if is_likely_invoice else 1

            doc_type_option = st.radio(
                "Document Schema Target:",
                ["Invoice (Vendor, Invoice No, Tax & Arithmetic Validation)", "Receipt (Store, Date, Address, Total)"],
                index=default_index,
                horizontal=True,
                help="Automatically inferred from document content. Toggle to select desired extraction schema and validation rules.",
            )
            doc_type_hint = "invoice" if "invoice" in doc_type_option.lower() else "receipt"

            with st.expander("Inspect Raw Ingested Text", expanded=False):
                st.text_area("Raw Text Layer", value=ingestion_res.text, height=180, disabled=True)

        st.divider()

        # Action: Trigger Extraction & Validation
        btn_extract = st.button("🚀 Extract & Validate Document", type="primary", use_container_width=True)

        # Automatic extraction trigger if document changed
        doc_id_key = f"{filename}_{ingestion_res.character_count}"
        should_run = btn_extract or (st.session_state["last_extracted_id"] != doc_id_key)

        if should_run:
            with st.spinner("Executing extraction pipeline and evaluating business rules..."):
                # Determine provider mode
                choice_lower = engine_choice.lower()
                if "auto" in choice_lower:
                    mode_key = "auto"
                elif "nvidia" in choice_lower:
                    mode_key = "nvidia"
                else:
                    mode_key = "baseline"

                extraction_res = extract_with_provider(
                    text=ingestion_res.text,
                    doc_type=doc_type_hint,
                    doc_id=filename,
                    mode=mode_key,
                    nvidia_api_key=active_nvidia_key,
                    nvidia_model=selected_nvidia_model,
                    image_bytes=ingestion_res.image_bytes,
                )

                # Execute Deterministic Validation
                validation_summary = validate_document(extraction_res.fields, doc_type=doc_type_hint)

                # Execute Review Decider
                review_decision = determine_review_routing(
                    fields=extraction_res.fields,
                    validation_summary=validation_summary,
                    doc_type=doc_type_hint,
                    raw_text=ingestion_res.text,
                )

                # Persist to Session State
                st.session_state["last_extracted_id"] = doc_id_key
                st.session_state["extraction_result"] = extraction_res
                st.session_state["validation_summary"] = validation_summary
                st.session_state["review_decision"] = review_decision
                st.session_state["manual_approved"] = False

        # Display Extraction Results if available
        if st.session_state["extraction_result"] is not None:
            ext_res: ExtractionResult = st.session_state["extraction_result"]
            val_sum: DocumentValidationSummary = st.session_state["validation_summary"]
            rev_dec: ReviewDecision = st.session_state["review_decision"]

            res_c1, res_c2 = st.columns([1, 1])

            with res_c1:
                st.markdown("#### Structured Extraction Output")
                if ext_res.source == "nvidia":
                    input_mode = ext_res.metadata.get("input_mode", "ocr_text_only") if hasattr(ext_res, "metadata") and ext_res.metadata else "ocr_text_only"
                    if input_mode == "multimodal_visual_text":
                        st.success(f"**Source Engine:** 🟢 NVIDIA NIM (`{ext_res.model_used}`) — 👁️ Multimodal Visual + OCR Input")
                    elif "super-vl" in (ext_res.model_used or "").lower():
                        st.info(f"**Source Engine:** 🟢 NVIDIA NIM (`{ext_res.model_used}`) — 📝 OCR-Text Input Mode (No visual input sent)")
                    else:
                        st.success(f"**Source Engine:** 🟢 NVIDIA NIM Hosted LLM (`{ext_res.model_used}`)")
                elif ext_res.source == "baseline_fallback":
                    if ext_res.error_message:
                        st.warning(f"⚠️ **Fallback Active:** Hosted LLM extraction was attempted but failed ({ext_res.error_message}). Successfully switched to deterministic baseline.")
                    else:
                        st.warning("⚠️ **Fallback Active:** Switched to deterministic baseline.")
                else:
                    st.info("ℹ️ **Source Engine:** 🛡️ Deterministic Regex Baseline (Offline Mode)")

                # Build fields table
                field_rows = []
                for k, v in ext_res.fields.items():
                    conf = rev_dec.field_confidences.get(k, 0.0)
                    evidence = rev_dec.evidence.get(k, "")
                    field_rows.append({
                        "Field": k,
                        "Extracted Value": str(v) if v is not None else "[Missing]",
                        "Confidence": f"{conf * 100:.1f}%",
                        "Evidence Rationale": evidence,
                    })

                fields_df = pd.DataFrame(field_rows)
                st.dataframe(fields_df, use_container_width=True, hide_index=True)

                # Download Button
                json_payload = {
                    "document_id": ext_res.doc_id,
                    "document_type": ext_res.doc_type,
                    "provider": ext_res.provider,
                    "source": ext_res.source,
                    "model": ext_res.model_used,
                    "review_status": rev_dec.status,
                    "overall_confidence": rev_dec.overall_confidence,
                    "extracted_fields": ext_res.fields,
                    "validation_rules_passed": f"{val_sum.passed_rules}/{val_sum.total_rules}",
                    "engine_notes": ext_res.error_message,
                }
                st.download_button(
                    label="💾 Download Extracted JSON",
                    data=json.dumps(json_payload, indent=2),
                    file_name=f"extracted_{filename}.json",
                    mime="application/json",
                )

            with res_c2:
                st.markdown("#### Ground Truth Comparison & Evaluation")
                if ground_truth_obj:
                    st.success(f"Paired Ground Truth Available: `{ground_truth_obj.doc_id}`")
                    gt_dict = ground_truth_obj.model_dump()

                    comp_rows = []
                    for f, extracted_val in ext_res.fields.items():
                        if f in gt_dict:
                            gt_val = gt_dict[f]
                            exact, norm, fuzz = match_field(extracted_val, gt_val, f)
                            comp_rows.append({
                                "Field": f,
                                "Extracted": str(extracted_val),
                                "Ground Truth": str(gt_val),
                                "Exact Match": "PASS" if exact else "FAIL",
                                "Normalized Match": "PASS" if norm else "FAIL",
                                "RapidFuzz": f"{fuzz:.1f}",
                            })

                    if comp_rows:
                        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

                    if getattr(ground_truth_obj, "injected_errors", None):
                        st.warning(f"⚠️ Deliberate Ground-Truth Edge Cases: {ground_truth_obj.injected_errors}")
                else:
                    st.info(
                        "No ground-truth annotation paired with this file. "
                        "Custom uploads are processed through zero-shot extraction and deterministic business-rule validation."
                    )

# =============================================================================
# TAB 2: VALIDATION & REVIEW QUEUE
# =============================================================================
with tab_validate:
    st.markdown("### 📋 Business Rule Validation & Human Review Queue")

    if st.session_state["review_decision"] is None:
        st.info("No document has been processed yet. Go to **Tab 1** and click **Extract & Validate Document**.")
    else:
        rev_dec: ReviewDecision = st.session_state["review_decision"]
        val_sum: DocumentValidationSummary = st.session_state["validation_summary"]
        ext_res: ExtractionResult = st.session_state["extraction_result"]

        # Review Status Callout Banner
        status_c1, status_c2, status_c3 = st.columns([2, 1, 1])

        with status_c1:
            if st.session_state["manual_approved"]:
                st.markdown(
                    '<div class="status-badge-approved">✅ OVERRIDDEN & APPROVED BY HUMAN REVIEWER</div>',
                    unsafe_allow_html=True,
                )
                st.success("Document has been verified and marked approved for downstream ERP / AP export.")
            elif rev_dec.status == ReviewStatus.AUTO_APPROVED:
                st.markdown(
                    '<div class="status-badge-approved">✅ AUTO APPROVED — HIGH CONFIDENCE</div>',
                    unsafe_allow_html=True,
                )
                st.success("All arithmetic and chronological business rules passed. Safe for automated straight-through processing.")
            elif rev_dec.status == ReviewStatus.FLAGGED_WITH_WARNINGS:
                st.markdown(
                    '<div class="status-badge-warning">⚠️ FLAGGED WITH WARNINGS — OPTIONAL SPOT CHECK</div>',
                    unsafe_allow_html=True,
                )
                st.warning("All critical arithmetic rules passed, but non-critical warnings or moderate confidence require quick verification.")
            else:
                st.markdown(
                    '<div class="status-badge-review">🚨 HUMAN REVIEW REQUIRED — VALIDATION FAILED</div>',
                    unsafe_allow_html=True,
                )
                st.error("Deterministic business rule violation detected (e.g., arithmetic mismatch, inverted dates, or missing total).")

        with status_c2:
            st.metric(
                label="Overall Confidence",
                value=f"{rev_dec.overall_confidence * 100:.1f}%",
                delta="Evidence Score",
            )

        with status_c3:
            st.metric(
                label="Rules Validated",
                value=f"{val_sum.passed_rules} / {val_sum.total_rules}",
                delta=f"{len(val_sum.errors)} Errors" if val_sum.errors else "0 Errors",
                delta_color="inverse" if val_sum.errors else "normal",
            )

        st.divider()

        # Decision Reasons & Rule Diagnostics
        v_col_left, v_col_right = st.columns([1, 1])

        with v_col_left:
            st.markdown("#### Decider Rationales")
            for r in rev_dec.reasons:
                if "Error" in r or "mismatch" in r.lower() or "violation" in r.lower():
                    st.error(f"• {r}")
                elif "Warning" in r or "moderate" in r.lower():
                    st.warning(f"• {r}")
                else:
                    st.success(f"• {r}")

            st.markdown("#### Business Rule Checklist")
            rule_rows = []
            for rule in val_sum.rule_results:
                status_icon = "✅ Passed" if rule.passed else ("❌ Failed" if rule.severity == "ERROR" else "⚠️ Warning")
                rule_rows.append({
                    "Rule Identifier": rule.rule_name,
                    "Status": status_icon,
                    "Severity": rule.severity,
                    "Diagnostic Explanation": rule.message,
                })
            st.dataframe(pd.DataFrame(rule_rows), use_container_width=True, hide_index=True)

        with v_col_right:
            st.markdown("#### ✏️ Human-in-the-Loop Review & Override Panel")
            st.caption("Reviewers can inspect, modify erroneous fields, and trigger immediate re-validation.")

            with st.form("human_review_form"):
                edited_fields = {}
                for k, v in ext_res.fields.items():
                    val_str = str(v) if v is not None else ""
                    edited_fields[k] = st.text_input(f"Field: `{k}`", value=val_str)

                form_c1, form_c2 = st.columns(2)
                with form_c1:
                    reval_submitted = st.form_submit_button("🔄 Re-Validate Changes")
                with form_c2:
                    approve_submitted = st.form_submit_button("✅ Mark Approved & Export")

                if reval_submitted:
                    # Cast numeric fields
                    cleaned_dict = {}
                    for k, v in edited_fields.items():
                        if v.strip() == "":
                            cleaned_dict[k] = None
                        elif any(num_key in k for num_key in ["total", "subtotal", "tax", "discount"]):
                            try:
                                cleaned_dict[k] = float(v.replace("$", "").replace(",", "").strip())
                            except ValueError:
                                cleaned_dict[k] = v
                        else:
                            cleaned_dict[k] = v.strip()

                    new_val_summary = validate_document(cleaned_dict, doc_type=ext_res.doc_type)
                    new_rev_decision = determine_review_routing(
                        fields=cleaned_dict,
                        validation_summary=new_val_summary,
                        doc_type=ext_res.doc_type,
                        raw_text=ext_res.raw_text,
                    )
                    st.session_state["extraction_result"].fields = cleaned_dict
                    st.session_state["validation_summary"] = new_val_summary
                    st.session_state["review_decision"] = new_rev_decision
                    st.session_state["manual_approved"] = False
                    st.rerun()

                if approve_submitted:
                    st.session_state["manual_approved"] = True
                    st.rerun()

# =============================================================================
# TAB 3: BENCHMARK ANALYTICS
# =============================================================================
with tab_benchmark:
    st.markdown("### 📊 Empirical Performance Benchmark & Evaluation")
    st.caption("Demonstrating empirical baseline measurements on SROIE development data and live edge-case detection.")

    b_metric1, b_metric2, b_metric3, b_metric4 = st.columns(4)

    # Load baseline summary CSV from disk
    baseline_csv_path = RESULTS_DIR / "dev_baseline_summary.csv"
    if baseline_csv_path.exists():
        dev_bench_df = pd.read_csv(baseline_csv_path)
        macro_row = dev_bench_df[dev_bench_df["Field"] == "MACRO AVERAGE"].iloc[0]
        macro_em = float(macro_row["Exact Match (%)"])
        macro_nm = float(macro_row["Normalized Match (%)"])
        macro_fuzz = float(macro_row["Avg RapidFuzz Score"])

        with b_metric1:
            st.metric("Macro Exact Match", f"{macro_em:.2f}%", delta="SROIE Dev Set (N=349)")
        with b_metric2:
            st.metric("Macro Normalized Match", f"{macro_nm:.2f}%", delta="+29.79% via Normalization")
        with b_metric3:
            st.metric("Avg RapidFuzz Score", f"{macro_fuzz:.2f} / 100", delta="Token Similarity")
        with b_metric4:
            st.metric("Held-Out Test Records", "88 Sealed", delta="Sealed for Evaluation")

        st.divider()

        # Interactive Comparison Chart
        chart_c1, chart_c2 = st.columns([1, 1])

        with chart_c1:
            st.markdown("#### SROIE Development Set Baseline (N = 349)")
            st.dataframe(dev_bench_df, use_container_width=True, hide_index=True)

            with st.expander("Diagnostic Failure Analysis"):
                st.markdown(
                    """
                    - **`company` (12.32% EM / 39.26% NM):** Confused by header logos, shopping mall titles (e.g. *IOI City Mall*), and tax registration tokens.
                    - **`date` (57.88% EM / 68.19% NM):** Strongest heuristic field. Standard regex captures two-thirds of valid dates.
                    - **`address` (0.00% EM / 69.05% NM):** Multi-line OCR splits addresses across 3–5 lines, resulting in 0% character-exact string equality, but 69.05% token normalized overlap.
                    - **`total` (39.54% EM / 52.44% NM):** Multiple currency figures (Subtotal, GST, Cash Tendered, Change Due) create heuristic ambiguity.
                    """
                )

        with chart_c2:
            st.markdown("#### Field-Level Match Distribution")
            chart_data = dev_bench_df[dev_bench_df["Field"] != "MACRO AVERAGE"].copy()
            fig = go.Figure(
                data=[
                    go.Bar(name="Exact Match (%)", x=chart_data["Field"], y=chart_data["Exact Match (%)"], marker_color="#4361ee"),
                    go.Bar(name="Normalized Match (%)", x=chart_data["Field"], y=chart_data["Normalized Match (%)"], marker_color="#4cc9f0"),
                    go.Bar(name="Avg RapidFuzz Score", x=chart_data["Field"], y=chart_data["Avg RapidFuzz Score"], marker_color="#f72585"),
                ]
            )
            fig.update_layout(
                barmode="group",
                yaxis_title="Score (%)",
                xaxis_title="Receipt Field",
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Baseline evaluation CSV (`results/dev_baseline_summary.csv`) not found on disk.")

    st.divider()

    # Live Synthetic Edge-Case Benchmark Suite
    st.markdown("#### 🧪 Live Synthetic Testbed Evaluation (10 Documents)")
    st.caption("Instantly verifies deterministic business-rule detection on local synthetic edge cases without re-running expensive OCR.")

    if st.button("▶️ Run Synthetic Testbed Benchmark", type="secondary"):
        with st.spinner("Evaluating deterministic business rules on synthetic testbed..."):
            test_results = []
            all_samples = sorted(list(SAMPLE_INVOICES_DIR.glob("*.pdf")) + list(SAMPLE_RECEIPTS_DIR.glob("*.pdf")))

            for p in all_samples:
                gt_path = GROUND_TRUTH_DIR / f"{p.stem}.json"
                if not gt_path.exists():
                    continue

                gt_obj = load_ground_truth(gt_path)
                ing_res = ingest_document(p, filename=p.name)

                # Run baseline extraction
                base_ext = RegexBaselineExtractor()
                ext_dict = base_ext.extract(ing_res.text, doc_type=gt_obj.doc_type)

                # Validate
                val_res = validate_document(ext_dict, doc_type=gt_obj.doc_type)
                dec = determine_review_routing(ext_dict, val_res, doc_type=gt_obj.doc_type)

                injected = getattr(gt_obj, "injected_errors", [])
                expected_valid = getattr(gt_obj, "expected_valid", True)

                # Check if detected properly
                flagged_correctly = (not val_res.is_valid) if not expected_valid else val_res.is_valid

                test_results.append({
                    "Document ID": gt_obj.doc_id,
                    "Type": gt_obj.doc_type,
                    "Expected Valid": "YES" if expected_valid else "NO",
                    "Validation Passed": "YES" if val_res.is_valid else "NO",
                    "Routing Decision": dec.status,
                    "Edge Cases": ", ".join(injected) if injected else "None (Clean)",
                    "Testbed Result": "✅ PASS" if flagged_correctly else "⚠️ MISMATCH",
                })

            res_df = pd.DataFrame(test_results)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            pass_count = sum(1 for r in test_results if "PASS" in r["Testbed Result"])
            st.success(f"Edge-Case Detection Rate: **{pass_count} / {len(test_results)} ({pass_count/len(test_results)*100:.1f}%)** across synthetic test cases.")
