"""Unit tests verifying synthetic testbed generation and edge-case invariants."""

from src.config import SAMPLE_INVOICES_DIR, SAMPLE_RECEIPTS_DIR, GROUND_TRUTH_DIR
from src.schemas import load_ground_truth, InvoiceGroundTruth, ReceiptGroundTruth
from data.generate_synthetic import generate_all_synthetic_documents


def test_synthetic_generation_execution():
    """Verify generator runs deterministically and creates the full suite."""
    summary = generate_all_synthetic_documents()
    assert summary["invoices_generated"] == 7
    assert summary["receipts_generated"] == 3
    assert summary["total_documents"] == 10


def test_generated_pdf_files_exist_and_readable():
    """Verify all generated PDFs exist on disk and contain valid non-empty data."""
    for inv_num in range(1, 8):
        matching = list(SAMPLE_INVOICES_DIR.glob(f"inv_synth_0{inv_num}_*.pdf"))
        assert len(matching) == 1, f"Missing invoice PDF for test index 0{inv_num}"
        assert matching[0].stat().st_size > 500, f"Invoice PDF {matching[0]} is too small"

    for rcpt_num in range(1, 4):
        matching = list(SAMPLE_RECEIPTS_DIR.glob(f"rcpt_synth_0{rcpt_num}_*.pdf"))
        assert len(matching) == 1, f"Missing receipt PDF for test index 0{rcpt_num}"
        assert matching[0].stat().st_size > 500, f"Receipt PDF {matching[0]} is too small"


def test_edge_case_arithmetic_mismatch():
    """Verify arithmetic mismatch error annotation."""
    gt_file = GROUND_TRUTH_DIR / "inv_synth_03_arithmetic_mismatch.json"
    gt = load_ground_truth(gt_file)
    assert isinstance(gt, InvoiceGroundTruth)
    assert gt.expected_valid is False
    assert "arithmetic_mismatch" in gt.injected_errors
    # Verify the math discrepancy
    assert gt.subtotal == 300.0
    assert gt.tax_amount == 30.0
    assert gt.total_amount == 375.0  # 300 + 30 != 375


def test_edge_case_missing_total():
    """Verify missing total amount annotation."""
    gt_file = GROUND_TRUTH_DIR / "inv_synth_04_missing_total.json"
    gt = load_ground_truth(gt_file)
    assert isinstance(gt, InvoiceGroundTruth)
    assert gt.expected_valid is False
    assert "missing_total_amount" in gt.injected_errors
    assert gt.total_amount is None


def test_edge_case_inverted_chronology():
    """Verify due date precedes invoice date."""
    gt_file = GROUND_TRUTH_DIR / "inv_synth_05_chronology_inverted.json"
    gt = load_ground_truth(gt_file)
    assert isinstance(gt, InvoiceGroundTruth)
    assert gt.expected_valid is False
    assert "due_date_before_invoice_date" in gt.injected_errors
    assert gt.invoice_date > gt.due_date  # 2026-11-20 > 2026-10-15


def test_edge_case_non_positive_total():
    """Verify non-positive total amount annotation."""
    gt_file = GROUND_TRUTH_DIR / "inv_synth_06_non_positive_total.json"
    gt = load_ground_truth(gt_file)
    assert isinstance(gt, InvoiceGroundTruth)
    assert gt.expected_valid is False
    assert "non_positive_total_amount" in gt.injected_errors
    assert gt.total_amount <= 0.0


def test_edge_case_missing_vendor():
    """Verify omitted vendor name annotation."""
    gt_file = GROUND_TRUTH_DIR / "inv_synth_07_missing_vendor.json"
    gt = load_ground_truth(gt_file)
    assert isinstance(gt, InvoiceGroundTruth)
    assert gt.expected_valid is False
    assert "missing_vendor_name" in gt.injected_errors
    assert gt.vendor_name is None


def test_receipt_sroie_schema_compliance():
    """Verify synthetic receipts adhere strictly to the 4 target SROIE fields."""
    gt_file = GROUND_TRUTH_DIR / "rcpt_synth_01_valid_standard.json"
    gt = load_ground_truth(gt_file)
    assert isinstance(gt, ReceiptGroundTruth)
    assert gt.company == "CYNBIT COFFEE ROASTERS"
    assert gt.date == "2026-10-18"
    assert "JAIPUR" in gt.address
    assert gt.total == 16.50
