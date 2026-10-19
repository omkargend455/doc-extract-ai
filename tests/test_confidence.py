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


def test_slack_invoice_zero_tax_confidence_and_routing():
    """Verify confidence calculation and auto-approval for the Slack invoice example.

    Ground truth:
      - vendor_name: Slack
      - invoice_number: 1223113
      - invoice_date: 2023-12-01
      - due_date: 2024-01-15
      - subtotal: 1725.00
      - tax_amount: 0.00
      - total_amount: 1725.00
      - currency: USD
    """
    slack_raw_text = """
    slack INVOICE
    Slack
    500 Howard Street, San Francisco, CA 94105
    Date: Dec 1, 2023
    Due Date: Jan 15, 2024
    Invoice #: 1223113
    Balance Due: $1,725.00
    Subtotal $1,725.00
    Tax (0%) $0.00
    Total $1,725.00
    """
    slack_fields = {
        "vendor_name": "Slack",
        "invoice_number": "1223113",
        "invoice_date": "2023-12-01",
        "due_date": "2024-01-15",
        "subtotal": 1725.00,
        "tax_amount": 0.00,
        "sgst_amount": None,
        "cgst_amount": None,
        "igst_amount": None,
        "discount_amount": 0.00,
        "total_amount": 1725.00,
        "currency": "USD",
    }

    val_summary = validate_document(slack_fields, doc_type="invoice")
    assert val_summary.is_valid
    assert len(val_summary.errors) == 0

    decision = determine_review_routing(
        fields=slack_fields,
        validation_summary=val_summary,
        doc_type="invoice",
        raw_text=slack_raw_text,
    )

    # 1. Routing must be AUTO_APPROVED with high overall confidence
    assert decision.status == ReviewStatus.AUTO_APPROVED
    assert not decision.requires_human_attention
    assert decision.overall_confidence >= 0.85

    # 2. tax_amount must have high confidence verified by arithmetic balance (not 0.0 or 0.25)
    assert "tax_amount" in decision.field_confidences
    assert decision.field_confidences["tax_amount"] >= 0.95
    assert "verified mathematically" in decision.evidence["tax_amount"]
    assert "Zero tax" in decision.evidence["tax_amount"]

    # 3. Jurisdictional taxes for US invoice must be recognized as not applicable, not missing failures
    assert "sgst_amount" in decision.field_confidences
    assert decision.field_confidences["sgst_amount"] == 1.0
    assert "not applicable" in decision.evidence["sgst_amount"].lower()

    assert "cgst_amount" in decision.field_confidences
    assert decision.field_confidences["cgst_amount"] == 1.0
    assert "not applicable" in decision.evidence["cgst_amount"].lower()

    assert "igst_amount" in decision.field_confidences
    assert decision.field_confidences["igst_amount"] == 1.0
    assert "not applicable" in decision.evidence["igst_amount"].lower()

    # 4. Currency and discount must be present in field confidences
    assert "currency" in decision.field_confidences
    assert decision.field_confidences["currency"] >= 0.90
    assert "discount_amount" in decision.field_confidences
    assert decision.field_confidences["discount_amount"] >= 0.90


def test_zero_tax_standalone_confidence_grounded_in_evidence():
    """Verify zero tax confidence is evidence-grounded and not artificially inflated."""
    # Case A: Explicit 0% tax notation in document text -> high grounded confidence
    score_with_text, exp_with_text = compute_field_confidence(
        field_name="tax_amount",
        value=0.0,
        doc_type="invoice",
        validation_summary=None,
        raw_text="Item A: $100.00\nTax (0%): $0.00\nTotal: $100.00",
    )
    assert score_with_text >= 0.85
    assert "corroborated" in exp_with_text.lower()

    # Case B: Isolated zero without arithmetic check or explicit text -> moderate score, not inflated
    score_no_text, exp_no_text = compute_field_confidence(
        field_name="tax_amount",
        value=0.0,
        doc_type="invoice",
        validation_summary=None,
        raw_text="Item A: $100.00\nTotal: $100.00",
    )
    assert 0.65 <= score_no_text <= 0.75
    assert "lacks arithmetic" in exp_no_text.lower()

    # Case C: Missing tax is 0.0 confidence (not confused with explicit zero)
    score_missing, exp_missing = compute_field_confidence(
        field_name="tax_amount",
        value=None,
        doc_type="invoice",
    )
    assert score_missing == 0.0
    assert "missing" in exp_missing.lower()


def test_negative_tax_and_amounts_penalized_for_accounting_violation():
    """Verify negative monetary amounts receive accounting violation penalty."""
    score_neg_tax, exp_neg_tax = compute_field_confidence(
        field_name="tax_amount",
        value=-15.00,
        doc_type="invoice",
    )
    assert score_neg_tax == 0.25
    assert "negative" in exp_neg_tax.lower()

    score_neg_subtotal, exp_neg_subtotal = compute_field_confidence(
        field_name="subtotal",
        value=-100.00,
        doc_type="invoice",
    )
    assert score_neg_subtotal == 0.25
    assert "negative" in exp_neg_subtotal.lower()

