"""Unit tests for deterministic field and business-rule validation (src/validators.py)."""

import pytest
from src.schemas import DocumentValidationSummary, ValidationRuleResult, InvoiceGroundTruth, ReceiptGroundTruth
from src.validators import (
    validate_required_fields,
    validate_positive_amounts,
    validate_date_plausibility,
    validate_date_chronology,
    validate_arithmetic_consistency,
    validate_document,
    parse_date_to_iso,
)


def test_parse_date_to_iso():
    assert parse_date_to_iso("2024-03-25") == "2024-03-25"
    assert parse_date_to_iso("25/03/2024") == "2024-03-25"
    assert parse_date_to_iso("25-03-2024") == "2024-03-25"
    assert parse_date_to_iso("25.03.2024") == "2024-03-25"
    assert parse_date_to_iso("25 Mar 2024") == "2024-03-25"
    assert parse_date_to_iso("March 25, 2024") == "2024-03-25"
    assert parse_date_to_iso("invalid_date_token") is None
    assert parse_date_to_iso("") is None
    assert parse_date_to_iso(None) is None


def test_required_fields_receipt():
    valid_receipt = {"company": "Lotus Hypermarket", "date": "2024-01-15", "total": 45.20}
    results = validate_required_fields(valid_receipt, doc_type="receipt")
    assert all(r.passed for r in results)

    missing_company = {"date": "2024-01-15", "total": 45.20}
    results = validate_required_fields(missing_company, doc_type="receipt")
    failed = [r for r in results if not r.passed]
    assert len(failed) == 1
    assert failed[0].field == "company"


def test_required_fields_invoice():
    valid_inv = {"vendor_name": "Acme Corp", "invoice_date": "2024-02-01", "total_amount": 1100.0}
    results = validate_required_fields(valid_inv, doc_type="invoice")
    assert all(r.passed for r in results)

    missing_vendor = {"invoice_date": "2024-02-01", "total_amount": 1100.0}
    results = validate_required_fields(missing_vendor, doc_type="invoice")
    failed = [r for r in results if not r.passed]
    assert len(failed) == 1
    assert failed[0].field == "vendor_name"


def test_positive_amounts_validation():
    # Valid positive total
    res = validate_positive_amounts({"total": 52.80}, doc_type="receipt")
    assert res[0].passed

    # Zero total (invalid)
    res_zero = validate_positive_amounts({"total": 0.0}, doc_type="receipt")
    assert not res_zero[0].passed
    assert res_zero[0].severity == "ERROR"

    # Negative total (invalid)
    res_neg = validate_positive_amounts({"total_amount": -15.50}, doc_type="invoice")
    assert not res_neg[0].passed
    assert res_neg[0].severity == "ERROR"

    # Negative subtotal (invalid)
    res_sub = validate_positive_amounts({"total_amount": 100.0, "subtotal": -10.0}, doc_type="invoice")
    failed_sub = [r for r in res_sub if not r.passed]
    assert len(failed_sub) == 1
    assert failed_sub[0].field == "subtotal"


def test_date_plausibility():
    # Valid calendar date in range
    res = validate_date_plausibility({"invoice_date": "2023-11-20"}, doc_type="invoice")
    assert all(r.passed for r in res)

    # Unparseable date token
    res_bad = validate_date_plausibility({"date": "99-99-9999"}, doc_type="receipt")
    assert not res_bad[0].passed
    assert res_bad[0].severity == "ERROR"

    # Date with year outside 2000-2035 range
    res_old = validate_date_plausibility({"date": "1985-05-12"}, doc_type="receipt")
    failed = [r for r in res_old if not r.passed]
    assert len(failed) == 1
    assert failed[0].field == "date"


def test_date_chronology_invoice():
    # Chronology valid: invoice_date <= due_date
    valid_dates = {"invoice_date": "2024-05-01", "due_date": "2024-05-31"}
    res = validate_date_chronology(valid_dates)
    assert res[0].passed

    # Chronology inverted: invoice_date > due_date
    inverted_dates = {"invoice_date": "2024-06-15", "due_date": "2024-05-01"}
    res_inverted = validate_date_chronology(inverted_dates)
    assert not res_inverted[0].passed
    assert res_inverted[0].severity == "ERROR"
    assert "invoice date" in res_inverted[0].message and "due date" in res_inverted[0].message

    # Skipped when either date is missing (no false alarms)
    missing_due = {"invoice_date": "2024-05-01"}
    res_missing = validate_date_chronology(missing_due)
    assert res_missing[0].passed
    assert res_missing[0].severity == "INFO"


def test_arithmetic_consistency_complete():
    # Valid: subtotal (100) + tax (10) - discount (5) == total (105)
    data = {
        "subtotal": 100.0,
        "tax_amount": 10.0,
        "discount_amount": 5.0,
        "total_amount": 105.0,
    }
    res = validate_arithmetic_consistency(data)
    assert res[0].passed
    assert res[0].details["delta"] == 0.0

    # Arithmetic mismatch: calculated 105 != stated 120
    mismatch = {
        "subtotal": 100.0,
        "tax_amount": 10.0,
        "discount_amount": 5.0,
        "total_amount": 120.0,
    }
    res_mismatch = validate_arithmetic_consistency(mismatch)
    assert not res_mismatch[0].passed
    assert res_mismatch[0].severity == "ERROR"
    assert res_mismatch[0].details["delta"] == 15.0


def test_arithmetic_consistency_no_inference_of_missing_components():
    """Verify that arithmetic check is strictly skipped if subtotal or total is absent."""
    # Subtotal missing -> skipped, does NOT assume subtotal is 0
    data_no_subtotal = {"tax_amount": 10.0, "total_amount": 110.0}
    res = validate_arithmetic_consistency(data_no_subtotal)
    assert res[0].passed
    assert res[0].severity == "INFO"
    assert "skipped" in res[0].message.lower()

    # Total missing -> skipped
    data_no_total = {"subtotal": 100.0, "tax_amount": 10.0}
    res2 = validate_arithmetic_consistency(data_no_total)
    assert res2[0].passed
    assert res2[0].severity == "INFO"


def test_arithmetic_consistency_rounding_tolerance():
    # Minor floating point rounding (within +- 0.05) should pass
    data = {
        "subtotal": 33.33,
        "tax_amount": 3.33,
        "total_amount": 36.65,  # diff is 0.01
    }
    res = validate_arithmetic_consistency(data)
    assert res[0].passed


def test_validate_document_end_to_end():
    # 1. Clean valid invoice
    valid_doc = {
        "doc_type": "invoice",
        "vendor_name": "Apex Office Supplies",
        "invoice_number": "INV-2024-001",
        "invoice_date": "2024-03-01",
        "due_date": "2024-03-31",
        "subtotal": 500.0,
        "tax_amount": 50.0,
        "discount_amount": 0.0,
        "total_amount": 550.0,
    }
    summary = validate_document(valid_doc)
    assert summary.is_valid
    assert len(summary.errors) == 0

    # 2. Corrupted invoice with multiple errors
    corrupted_doc = {
        "doc_type": "invoice",
        # missing vendor_name
        "invoice_date": "2024-05-10",
        "due_date": "2024-04-01",  # inverted chronology
        "subtotal": 200.0,
        "tax_amount": 20.0,
        "total_amount": -50.0,      # negative total
    }
    corrupted_summary = validate_document(corrupted_doc)
    assert not corrupted_summary.is_valid
    error_rules = {e.rule_name for e in corrupted_summary.errors}
    assert "required_field_vendor_name" in error_rules
    assert "positive_total_amount" in error_rules
    assert "date_chronology" in error_rules
