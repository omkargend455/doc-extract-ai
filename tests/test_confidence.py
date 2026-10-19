"""Unit tests for composite confidence scoring and review routing (src/confidence.py)."""

import pytest
from src.schemas import ReviewStatus
from src.validators import validate_document
from src.confidence import (
    compute_field_confidence,
    compute_composite_confidence,
    determine_review_routing,
)


def test_field_confidence_signals():
    # 1. Valid date signal
    score_date, exp_date = compute_field_confidence("date", "2024-04-12", "receipt")
    assert score_date >= 0.90
    assert "calendar format" in exp_date

    # 2. Invalid date format signal
    score_bad_date, exp_bad_date = compute_field_confidence("date", "random-text", "receipt")
    assert score_bad_date < 0.50

    # 3. Missing field signal
    score_missing, exp_missing = compute_field_confidence("company", None, "receipt")
    assert score_missing == 0.0

    # 4. Clean vendor name signal
    score_vendor, exp_vendor = compute_field_confidence("vendor_name", "Petronas Gas Berhad", "invoice")
    assert score_vendor >= 0.80

    # 5. Suspiciously short vendor name
    score_short, exp_short = compute_field_confidence("company", "AB", "receipt")
    assert score_short <= 0.40


def test_auto_approval_routing():
    clean_receipt = {
        "company": "KFC Holdings (M) Bhd",
        "date": "2024-02-14",
        "address": "No 15, Jalan Bukit Bintang, 55100 Kuala Lumpur",
        "total": 38.50,
    }
    validation_summary = validate_document(clean_receipt, doc_type="receipt")
    decision = determine_review_routing(clean_receipt, validation_summary, doc_type="receipt")

    assert decision.status == ReviewStatus.AUTO_APPROVED
    assert not decision.requires_human_attention
    assert decision.overall_confidence >= 0.85
    assert len(decision.reasons) > 0


def test_human_review_required_on_arithmetic_mismatch():
    """Verify that an arithmetic calculation failure immediately triggers HUMAN_REVIEW_REQUIRED."""
    corrupted_invoice = {
        "doc_type": "invoice",
        "vendor_name": "Tech Supplies Inc",
        "invoice_number": "INV-9901",
        "invoice_date": "2024-01-10",
        "due_date": "2024-02-10",
        "subtotal": 1000.0,
        "tax_amount": 100.0,
        "discount_amount": 0.0,
        "total_amount": 800.0,  # Arithmetic mismatch (expected 1100.0)
    }
    validation_summary = validate_document(corrupted_invoice, doc_type="invoice")
    decision = determine_review_routing(corrupted_invoice, validation_summary, doc_type="invoice")

    assert decision.status == ReviewStatus.HUMAN_REVIEW_REQUIRED
    assert decision.requires_human_attention
    assert any("Arithmetic mismatch" in r for r in decision.reasons)


def test_human_review_required_on_missing_total():
    missing_total_invoice = {
        "doc_type": "invoice",
        "vendor_name": "Logistics Express",
        "invoice_date": "2024-03-05",
        "total_amount": None,
    }
    validation_summary = validate_document(missing_total_invoice, doc_type="invoice")
    decision = determine_review_routing(missing_total_invoice, validation_summary, doc_type="invoice")

    assert decision.status == ReviewStatus.HUMAN_REVIEW_REQUIRED
    assert decision.requires_human_attention
    assert any("required" in r.lower() for r in decision.reasons)


def test_human_review_required_on_chronology_inversion():
    inverted_invoice = {
        "doc_type": "invoice",
        "vendor_name": "Consulting Group",
        "invoice_date": "2024-06-30",
        "due_date": "2024-06-01",  # due date before invoice date
        "total_amount": 5000.0,
    }
    validation_summary = validate_document(inverted_invoice, doc_type="invoice")
    decision = determine_review_routing(inverted_invoice, validation_summary, doc_type="invoice")

    assert decision.status == ReviewStatus.HUMAN_REVIEW_REQUIRED
    assert decision.requires_human_attention
    assert any("chronology" in r.lower() for r in decision.reasons)


def test_evidence_grounding_transparency():
    """Verify that every field has an explicit human-readable evidence explanation."""
    receipt = {
        "company": "Giant Supermarket",
        "date": "2024-01-20",
        "address": "Lot 12 Jalan Sungai Buloh, 47000 Selangor",
        "total": 124.80,
    }
    validation_summary = validate_document(receipt, doc_type="receipt")
    decision = determine_review_routing(receipt, validation_summary, doc_type="receipt")

    for f in ["company", "date", "address", "total"]:
        assert f in decision.evidence
        assert len(decision.evidence[f]) > 10  # Non-trivial explanation
        assert f in decision.field_confidences
        assert 0.0 <= decision.field_confidences[f] <= 1.0
