"""Unit tests for Pydantic ground-truth schemas and parsing utilities."""

import json
import pytest
from src.schemas import (
    ReceiptGroundTruth,
    InvoiceGroundTruth,
    load_ground_truth,
    save_ground_truth,
)
from src.config import GROUND_TRUTH_DIR


def test_receipt_ground_truth_total_normalization():
    """Verify numeric normalization for various string formats and edge cases."""
    # Float directly
    r1 = ReceiptGroundTruth(doc_id="r1", total=45.50)
    assert r1.total == 45.50

    # Currency symbols & spaces
    r2 = ReceiptGroundTruth(doc_id="r2", total="$ 123.45")
    assert r2.total == 123.45

    r3 = ReceiptGroundTruth(doc_id="r3", total="RM 89.90")
    assert r3.total == 89.90

    r4 = ReceiptGroundTruth(doc_id="r4", total="₹ 1,500.00")
    assert r4.total == 1500.00

    # None and empty strings
    r5 = ReceiptGroundTruth(doc_id="r5", total="")
    assert r5.total is None

    r6 = ReceiptGroundTruth(doc_id="r6", total=None)
    assert r6.total is None

    # Unparseable strings
    r7 = ReceiptGroundTruth(doc_id="r7", total="N/A")
    assert r7.total is None


def test_invoice_ground_truth_defaults_and_errors():
    """Verify invoice ground-truth instantiation and error metadata."""
    inv = InvoiceGroundTruth(
        doc_id="inv_test",
        vendor_name="Test Corp",
        subtotal=100.0,
        tax_amount=10.0,
        total_amount=110.0,
        expected_valid=True,
        injected_errors=[],
    )
    assert inv.doc_type == "invoice"
    assert inv.currency == "USD"
    assert inv.expected_valid is True
    assert inv.discount_amount == 0.0


def test_load_ground_truth_from_disk():
    """Verify that existing generated ground-truth files load and parse correctly."""
    inv_file = GROUND_TRUTH_DIR / "inv_synth_01_valid_standard.json"
    rcpt_file = GROUND_TRUTH_DIR / "rcpt_synth_01_valid_standard.json"

    assert inv_file.exists()
    assert rcpt_file.exists()

    inv = load_ground_truth(inv_file)
    assert isinstance(inv, InvoiceGroundTruth)
    assert inv.doc_id == "inv_synth_01_valid_standard"
    assert inv.total_amount == 220.0
    assert inv.expected_valid is True

    rcpt = load_ground_truth(rcpt_file)
    assert isinstance(rcpt, ReceiptGroundTruth)
    assert rcpt.doc_id == "rcpt_synth_01_valid_standard"
    assert rcpt.company == "CYNBIT COFFEE ROASTERS"
    assert rcpt.total == 16.50


def test_save_ground_truth_roundtrip(tmp_path):
    """Verify saving and reloading ground truth preserve data integrity."""
    test_gt = ReceiptGroundTruth(
        doc_id="temp_rcpt_01",
        company="Temporary Store",
        date="2026-10-19",
        address="123 Test Lane",
        total=99.99,
        notes="Roundtrip test",
    )
    target = tmp_path / "temp_rcpt_01.json"
    save_ground_truth(test_gt, target)

    loaded = load_ground_truth(target)
    assert isinstance(loaded, ReceiptGroundTruth)
    assert loaded.doc_id == "temp_rcpt_01"
    assert loaded.company == "Temporary Store"
    assert loaded.total == 99.99


def test_load_non_existent_file():
    """Verify FileNotFoundError on missing files."""
    with pytest.raises(FileNotFoundError):
        load_ground_truth("non_existent_file.json")
