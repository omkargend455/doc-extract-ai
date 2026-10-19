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


def test_slack_invoice_zero_tax_arithmetic_consistency():
    """Verify arithmetic validation preserves explicit zero tax for the Slack invoice example."""
    data = {
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
    res = validate_arithmetic_consistency(data)
    assert len(res) == 1
    assert res[0].passed is True
    assert res[0].severity == "INFO"
    assert res[0].details["subtotal"] == 1725.00
    assert res[0].details["tax_amount"] == 0.00
    assert res[0].details["tax_amount"] is not None
    assert res[0].details["expected_total"] == 1725.00
    assert res[0].details["stated_total"] == 1725.00
    assert res[0].details["delta"] == 0.0
    assert "Tax (0.00)" in res[0].message


def test_distinguish_explicit_zero_tax_from_missing_tax():
    """Verify the pipeline distinguishes explicitly declared zero tax from missing tax."""
    # Case 1: Explicit zero tax matching total
    zero_tax_doc = {
        "subtotal": 100.0,
        "tax_amount": 0.0,
        "total_amount": 100.0,
    }
    res_zero = validate_arithmetic_consistency(zero_tax_doc)
    assert res_zero[0].passed is True
    assert res_zero[0].details["tax_amount"] == 0.0
    assert "Tax (0.00)" in res_zero[0].message

    # Case 2: Subtotal != total with missing tax -> fails with missing breakdown error
    missing_tax_mismatch = {
        "subtotal": 100.0,
        "tax_amount": None,
        "total_amount": 118.0,
    }
    res_missing = validate_arithmetic_consistency(missing_tax_mismatch)
    assert res_missing[0].passed is False
    assert res_missing[0].severity == "ERROR"
    assert "tax breakdown is missing" in res_missing[0].message.lower()
    assert res_missing[0].details["tax_amount"] is None

    # Case 3: Subtotal == total with undeclared tax -> passes without fabricating zero tax
    undeclared_tax_match = {
        "subtotal": 100.0,
        "tax_amount": None,
        "total_amount": 100.0,
    }
    res_undec = validate_arithmetic_consistency(undeclared_tax_match)
    assert res_undec[0].passed is True
    assert res_undec[0].details["tax_amount"] is None


def test_us_invoice_null_jurisdictional_taxes_remain_none():
    """Verify non-GST invoices keep SGST/CGST/IGST strictly null without fabricating values."""
    slack_data = {
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
    summary = validate_document(slack_data, doc_type="invoice")
    assert summary.is_valid
    assert len(summary.errors) == 0

    # Ensure none of the rules failed or threw warnings about missing jurisdictional taxes
    assert not any("sgst" in r.rule_name.lower() for r in summary.errors)
    assert not any("cgst" in r.rule_name.lower() for r in summary.errors)
    assert not any("igst" in r.rule_name.lower() for r in summary.errors)

