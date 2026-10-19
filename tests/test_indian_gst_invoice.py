"""Regression tests for tax-inclusive Indian GST invoices (BTPL Distribution Private Limited case)."""

import pytest
from src.baseline_regex import RegexBaselineExtractor
from src.schemas import InvoiceGroundTruth, ReviewStatus
from src.validators import validate_arithmetic_consistency, validate_document
from src.confidence import determine_review_routing


BTPL_INVOICE_TEXT = """
Description Qty Gross Amount ₹ Discounts/Coupons Taxable value ₹ SGST/UTGST ₹ CGST ₹ Total ₹

SAC: 998599 Convenience Fee 1 149.00 0.00 126.28 11.36 11.36 149.00
1. [IMEI/Serial No: 351060966096413]
SGST/UTGST: 9.0 %
CGST: 9.0 %

SAC: 998599 Protect Promise Fee 1 299.00 0.00 253.38 22.81 22.81 299.00
1. [IMEI/Serial No: 351060966096413]
SGST/UTGST: 9.0 %
CGST: 9.0 %

Total 2 448.00 0.00 379.66 34.17 34.17 448.00

Grand Total ₹ 448.00

BTPL Distribution Private Limited
"""


def test_btpl_distribution_invoice_baseline_extraction():
    """Verify deterministic extraction of BTPL Distribution invoice regression case."""
    extractor = RegexBaselineExtractor()
    fields = extractor.extract(BTPL_INVOICE_TEXT, doc_type="invoice")

    # 1. Vendor Name
    assert fields["vendor_name"] == "BTPL Distribution Private Limited"

    # 2. Subtotal (Consistently defined as pre-tax taxable value)
    assert fields["subtotal"] == 379.66

    # 3. Tax Amount (Sum of SGST 34.17 + CGST 34.17)
    assert fields["tax_amount"] == 68.34
    assert fields["sgst_amount"] == 34.17
    assert fields["cgst_amount"] == 34.17

    # 4. Discounts and Total
    assert fields["discount_amount"] == 0.00
    assert fields["total_amount"] == 448.00
    assert fields["currency"] == "INR"

    # 5. Missing fields (image does not contain invoice date or number; must not be invented)
    assert fields["invoice_number"] is None
    assert fields["invoice_date"] is None


def test_btpl_distribution_invoice_validation_arithmetic():
    """Verify that BTPL invoice arithmetic validates Subtotal + Tax == Total without adding tax twice."""
    extractor = RegexBaselineExtractor()
    fields = extractor.extract(BTPL_INVOICE_TEXT, doc_type="invoice")

    res = validate_arithmetic_consistency(fields)
    assert len(res) == 1
    assert res[0].passed is True
    assert res[0].details["subtotal"] == 379.66
    assert res[0].details["tax_amount"] == 68.34
    assert res[0].details["expected_total"] == 448.00
    assert res[0].details["stated_total"] == 448.00
    assert res[0].details["delta"] == 0.0

    # Ensure printed grand total is tax-inclusive: expected total is 448.00, NEVER 448.00 + 68.34 = 516.34
    assert res[0].details["expected_total"] != 516.34


def test_line_item_tax_not_mistaken_for_invoice_total():
    """Verify extractor does not mistake an individual line item's tax (11.36 or 22.81) for the invoice total tax."""
    extractor = RegexBaselineExtractor()
    fields = extractor.extract(BTPL_INVOICE_TEXT, doc_type="invoice")

    # Line 1 has 11.36 and Line 2 has 22.81
    assert fields["tax_amount"] != 11.36
    assert fields["tax_amount"] != 22.81
    assert fields["tax_amount"] != 22.72  # 11.36 * 2
    assert fields["tax_amount"] != 45.62  # 22.81 * 2
    assert fields["tax_amount"] == 68.34  # Total row 34.17 + 34.17


def test_missing_tax_breakdown_flags_human_review():
    """Verify that when subtotal != total and tax breakdown is missing, document is flagged for human review."""
    incomplete_fields = {
        "vendor_name": "BTPL Distribution Private Limited",
        "subtotal": 379.66,
        "tax_amount": None,
        "discount_amount": 0.00,
        "total_amount": 448.00,
        "currency": "INR",
    }

    # Deterministic validator must fail with ERROR (not invent tax of 68.34)
    val_res = validate_arithmetic_consistency(incomplete_fields)
    assert not val_res[0].passed
    assert val_res[0].severity == "ERROR"
    assert "tax breakdown is missing" in val_res[0].message

    summary = validate_document(incomplete_fields, doc_type="invoice")
    assert not summary.is_valid
    assert any(e.rule_name == "arithmetic_consistency" for e in summary.errors)

    # Review decider must route to HUMAN_REVIEW_REQUIRED
    decision = determine_review_routing(
        fields=incomplete_fields,
        validation_summary=summary,
        doc_type="invoice",
    )
    assert decision.status == ReviewStatus.HUMAN_REVIEW_REQUIRED
    assert decision.requires_human_attention is True


def test_inconsistent_tax_breakdown_flags_human_review():
    """Verify that when tax breakdown is inconsistent with total, document is flagged for human review."""
    inconsistent_fields = {
        "vendor_name": "BTPL Distribution Private Limited",
        "subtotal": 379.66,
        "tax_amount": 34.17,  # Inconsistent: only SGST without CGST
        "discount_amount": 0.00,
        "total_amount": 448.00,
        "currency": "INR",
    }

    val_res = validate_arithmetic_consistency(inconsistent_fields)
    assert not val_res[0].passed
    assert val_res[0].severity == "ERROR"
    assert "Arithmetic mismatch" in val_res[0].message

    summary = validate_document(inconsistent_fields, doc_type="invoice")
    assert not summary.is_valid

    decision = determine_review_routing(
        fields=inconsistent_fields,
        validation_summary=summary,
        doc_type="invoice",
    )
    assert decision.status == ReviewStatus.HUMAN_REVIEW_REQUIRED
    assert decision.requires_human_attention is True


def test_subtotal_defined_as_pre_tax_taxable_value():
    """Verify InvoiceGroundTruth schema defines subtotal consistently as pre-tax taxable value."""
    gt = InvoiceGroundTruth(
        doc_id="test_btpl",
        vendor_name="BTPL Distribution Private Limited",
        subtotal=379.66,
        sgst_amount=34.17,
        cgst_amount=34.17,
        discount_amount=0.0,
        total_amount=448.00,
        currency="INR",
    )
    # Automatic post-init mapping of SGST + CGST to tax_amount
    assert gt.tax_amount == 68.34
    assert gt.subtotal == 379.66
    assert gt.total_amount == 448.00
    assert round(gt.subtotal + gt.tax_amount - gt.discount_amount, 2) == gt.total_amount


def test_generic_gst_invoice_no_hardcoding():
    """Verify generic extraction on a completely different Indian GST invoice (verifies no vendor/amount hardcoding)."""
    generic_gst_text = """
Description Qty Gross Amount ₹ Discounts Taxable value ₹ SGST/UTGST ₹ CGST ₹ Total ₹
IT Consulting 1 1180.00 0.00 1000.00 90.00 90.00 1180.00
Total 1 1180.00 0.00 1000.00 90.00 90.00 1180.00
Grand Total ₹ 1180.00
Omni Solutions Private Limited
"""
    extractor = RegexBaselineExtractor()
    fields = extractor.extract(generic_gst_text, doc_type="invoice")

    assert fields["vendor_name"] == "Omni Solutions Private Limited"
    assert fields["subtotal"] == 1000.00
    assert fields["sgst_amount"] == 90.00
    assert fields["cgst_amount"] == 90.00
    assert fields["tax_amount"] == 180.00
    assert fields["total_amount"] == 1180.00
    assert fields["currency"] == "INR"

    val_res = validate_arithmetic_consistency(fields)
    assert val_res[0].passed is True
    assert val_res[0].details["expected_total"] == 1180.00
