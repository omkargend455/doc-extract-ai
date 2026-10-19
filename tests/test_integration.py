"""End-to-end integration tests for DocExtract AI pipeline (ingest -> extract -> validate -> route)."""

import pandas as pd
from pathlib import Path
import pytest

from src.config import (
    SAMPLE_INVOICES_DIR,
    SAMPLE_RECEIPTS_DIR,
    GROUND_TRUTH_DIR,
    RESULTS_DIR,
)
from src.ingest import ingest_document
from src.nvidia_extractor import NvidiaExtractor
from src.validators import validate_document
from src.confidence import determine_review_routing
from src.schemas import ReviewStatus, load_ground_truth


def test_end_to_end_valid_invoice_pipeline():
    """Verify clean invoice progresses through ingestion, extraction, validation, and auto-approval."""
    pdf_path = SAMPLE_INVOICES_DIR / "inv_synth_01_valid_standard.pdf"
    assert pdf_path.exists(), "Sample invoice PDF must exist"

    # 1. Ingestion
    ingestion = ingest_document(pdf_path, filename=pdf_path.name)
    assert ingestion.character_count > 50
    assert "Acme" in ingestion.text or "Solutions" in ingestion.text or "Total" in ingestion.text

    # 2. Offline Extraction
    extractor = NvidiaExtractor(api_key="")
    extraction = extractor.extract(ingestion.text, doc_type="invoice", doc_id=pdf_path.name)
    assert extraction.source == "baseline_fallback"
    assert extraction.fields["total_amount"] is not None
    assert extraction.fields["total_amount"] > 0

    # 3. Business Rule Validation
    val_summary = validate_document(extraction.fields, doc_type="invoice")
    assert val_summary.is_valid, f"Valid invoice should pass validation, failed: {[e.message for e in val_summary.errors]}"

    # 4. Review Routing Decider
    decision = determine_review_routing(
        fields=extraction.fields,
        validation_summary=val_summary,
        doc_type="invoice",
        raw_text=ingestion.text,
    )
    assert decision.status == ReviewStatus.AUTO_APPROVED
    assert not decision.requires_human_attention
    assert decision.overall_confidence >= 0.85


def test_end_to_end_arithmetic_mismatch_triggers_human_review():
    """Verify that an arithmetic error in inv_synth_03 is caught and flagged for mandatory human review."""
    pdf_path = SAMPLE_INVOICES_DIR / "inv_synth_03_arithmetic_mismatch.pdf"
    assert pdf_path.exists()

    # Ingest and Extract
    ingestion = ingest_document(pdf_path, filename=pdf_path.name)
    extractor = NvidiaExtractor(api_key="")
    extraction = extractor.extract(ingestion.text, doc_type="invoice", doc_id=pdf_path.name)

    # Validate
    val_summary = validate_document(extraction.fields, doc_type="invoice")
    assert not val_summary.is_valid
    error_names = {e.rule_name for e in val_summary.errors}
    assert "arithmetic_consistency" in error_names

    # Route
    decision = determine_review_routing(
        fields=extraction.fields,
        validation_summary=val_summary,
        doc_type="invoice",
        raw_text=ingestion.text,
    )
    assert decision.status == ReviewStatus.HUMAN_REVIEW_REQUIRED
    assert decision.requires_human_attention
    assert any("mismatch" in r.lower() for r in decision.reasons)


def test_end_to_end_receipt_pipeline():
    """Verify standard receipt progresses through ingestion, extraction, and validation."""
    pdf_path = SAMPLE_RECEIPTS_DIR / "rcpt_synth_01_valid_standard.pdf"
    assert pdf_path.exists()

    ingestion = ingest_document(pdf_path, filename=pdf_path.name)
    extractor = NvidiaExtractor(api_key="")
    extraction = extractor.extract(ingestion.text, doc_type="receipt", doc_id=pdf_path.name)

    val_summary = validate_document(extraction.fields, doc_type="receipt")
    assert val_summary.is_valid

    decision = determine_review_routing(
        fields=extraction.fields,
        validation_summary=val_summary,
        doc_type="receipt",
        raw_text=ingestion.text,
    )
    assert decision.status in {ReviewStatus.AUTO_APPROVED, ReviewStatus.FLAGGED_WITH_WARNINGS}


def test_baseline_summary_csv_schema_and_integrity():
    """Verify dev_baseline_summary.csv contains real measured metrics."""
    csv_path = RESULTS_DIR / "dev_baseline_summary.csv"
    assert csv_path.exists(), "dev_baseline_summary.csv must exist"

    df = pd.read_csv(csv_path)
    assert "Field" in df.columns
    assert "Exact Match (%)" in df.columns
    assert "Normalized Match (%)" in df.columns
    assert "Avg RapidFuzz Score" in df.columns

    fields = set(df["Field"].tolist())
    assert {"company", "date", "address", "total", "MACRO AVERAGE"}.issubset(fields)

    macro_row = df[df["Field"] == "MACRO AVERAGE"].iloc[0]
    assert float(macro_row["Exact Match (%)"]) == 27.44
    assert float(macro_row["Normalized Match (%)"]) == 57.23
    assert float(macro_row["Avg RapidFuzz Score"]) == 71.82
