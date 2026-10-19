"""Deterministic synthetic dataset generator for DocExtract AI.

Generates digital PDF documents and paired JSON ground-truth files for:
  - Invoices (standard valid, discount, arithmetic mismatch, inverted dates, missing totals)
  - Receipts (SROIE-compatible 4-field schema: company, date, address, total)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import pymupdf

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import SAMPLE_INVOICES_DIR, SAMPLE_RECEIPTS_DIR, GROUND_TRUTH_DIR
from src.schemas import InvoiceGroundTruth, ReceiptGroundTruth, save_ground_truth


def create_pdf_from_text(lines: list[str], output_path: Path) -> Path:
    """Render structured text into a digital PDF with PyMuPDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # Standard A4: 595 x 842 points

    y = 60
    for line in lines:
        if line.startswith("==="):
            page.draw_line((50, y - 5), (545, y - 5), color=(0.6, 0.6, 0.6), width=1)
            y += 12
        elif line.startswith("# "):
            page.insert_text((50, y), line[2:], fontsize=18, fontname="helv", color=(0.1, 0.2, 0.5))
            y += 26
        elif line.startswith("## "):
            page.insert_text((50, y), line[3:], fontsize=12, fontname="hebo", color=(0.2, 0.2, 0.2))
            y += 18
        elif line.startswith("[B]"):
            page.insert_text((50, y), line[3:], fontsize=10, fontname="hebo", color=(0, 0, 0))
            y += 16
        else:
            page.insert_text((50, y), line, fontsize=10, fontname="helv", color=(0.1, 0.1, 0.1))
            y += 15

    doc.save(str(output_path))
    doc.close()
    return output_path


def generate_all_synthetic_documents() -> dict[str, int]:
    """Generate the full synthetic testbed (PDFs and paired JSON ground truth)."""
    invoices_generated = 0
    receipts_generated = 0

    # Ensure directories exist
    SAMPLE_INVOICES_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. SYNTHETIC INVOICES
    # -------------------------------------------------------------

    # Invoice 1: Standard Valid Invoice
    inv1_id = "inv_synth_01_valid_standard"
    inv1_lines = [
        "# TAX INVOICE",
        "=== separator ===",
        "[B]Vendor: Apex Office Supplies Ltd",
        "Address: 100 Commercial Way, Tech District, Boston MA",
        "Email: billing@apexsupplies.example.com",
        "",
        "Invoice Number: INV-2026-001",
        "Invoice Date: 2026-10-01",
        "Payment Due Date: 2026-10-31",
        "Currency: USD",
        "=== separator ===",
        "## Line Items:",
        "1. Premium Copy Paper (5 Reams) - Quantity: 2 @ $40.00 = $80.00",
        "2. Ergonomic Desk Chairs - Quantity: 1 @ $120.00 = $120.00",
        "=== separator ===",
        "Subtotal: $200.00",
        "Tax Amount (10%): $20.00",
        "Discount Amount: $0.00",
        "[B]Total Amount Due: $220.00",
        "=== separator ===",
        "Payment Terms: Net 30 days. Thank you for your business!",
    ]
    inv1_pdf = SAMPLE_INVOICES_DIR / f"{inv1_id}.pdf"
    create_pdf_from_text(inv1_lines, inv1_pdf)
    inv1_gt = InvoiceGroundTruth(
        doc_id=inv1_id,
        vendor_name="Apex Office Supplies Ltd",
        invoice_number="INV-2026-001",
        invoice_date="2026-10-01",
        due_date="2026-10-31",
        subtotal=200.00,
        tax_amount=20.00,
        discount_amount=0.00,
        total_amount=220.00,
        currency="USD",
        items=[
            {"description": "Premium Copy Paper", "quantity": 2, "unit_price": 40.0, "amount": 80.0},
            {"description": "Ergonomic Desk Chairs", "quantity": 1, "unit_price": 120.0, "amount": 120.0},
        ],
        expected_valid=True,
        injected_errors=[],
        raw_text="\n".join(inv1_lines),
    )
    save_ground_truth(inv1_gt, GROUND_TRUTH_DIR / f"{inv1_id}.json")
    invoices_generated += 1

    # Invoice 2: Valid Multi-item with Discount
    inv2_id = "inv_synth_02_valid_discount"
    inv2_lines = [
        "# INVOICE",
        "=== separator ===",
        "[B]Vendor: TechFlow Infrastructure Inc",
        "Address: 500 Silicon Way, Suite 400, Austin TX",
        "",
        "Invoice Number: INV-2026-042",
        "Invoice Date: 2026-10-05",
        "Payment Due Date: 2026-11-05",
        "Currency: USD",
        "=== separator ===",
        "## Summary of Services:",
        "1. Dedicated Cloud Instance (Monthly) - $300.00",
        "2. Automated Backup Storage - $150.00",
        "=== separator ===",
        "Subtotal: $450.00",
        "Tax Amount: $36.00",
        "Promotional Discount: $50.00",
        "[B]Total Amount Due: $436.00",
        "=== separator ===",
        "Calculation: $450.00 + $36.00 - $50.00 = $436.00",
    ]
    inv2_pdf = SAMPLE_INVOICES_DIR / f"{inv2_id}.pdf"
    create_pdf_from_text(inv2_lines, inv2_pdf)
    inv2_gt = InvoiceGroundTruth(
        doc_id=inv2_id,
        vendor_name="TechFlow Infrastructure Inc",
        invoice_number="INV-2026-042",
        invoice_date="2026-10-05",
        due_date="2026-11-05",
        subtotal=450.00,
        tax_amount=36.00,
        discount_amount=50.00,
        total_amount=436.00,
        currency="USD",
        items=[
            {"description": "Dedicated Cloud Instance", "quantity": 1, "unit_price": 300.0, "amount": 300.0},
            {"description": "Automated Backup Storage", "quantity": 1, "unit_price": 150.0, "amount": 150.0},
        ],
        expected_valid=True,
        injected_errors=[],
        raw_text="\n".join(inv2_lines),
    )
    save_ground_truth(inv2_gt, GROUND_TRUTH_DIR / f"{inv2_id}.json")
    invoices_generated += 1

    # Invoice 3: Deliberate Arithmetic Mismatch
    inv3_id = "inv_synth_03_arithmetic_mismatch"
    inv3_lines = [
        "# INVOICE (TAMPERED / MISMATCH)",
        "=== separator ===",
        "[B]Vendor: Horizon Logistics Group",
        "Invoice Number: INV-2026-088",
        "Invoice Date: 2026-10-12",
        "Payment Due Date: 2026-11-12",
        "=== separator ===",
        "Subtotal: $300.00",
        "Tax Amount: $30.00",
        "Discount: $0.00",
        "[B]Total Amount: $375.00",
        "=== separator ===",
        "Note: Arithmetic mismatch injected (Expected 330.00, printed 375.00)",
    ]
    inv3_pdf = SAMPLE_INVOICES_DIR / f"{inv3_id}.pdf"
    create_pdf_from_text(inv3_lines, inv3_pdf)
    inv3_gt = InvoiceGroundTruth(
        doc_id=inv3_id,
        vendor_name="Horizon Logistics Group",
        invoice_number="INV-2026-088",
        invoice_date="2026-10-12",
        due_date="2026-11-12",
        subtotal=300.00,
        tax_amount=30.00,
        discount_amount=0.00,
        total_amount=375.00,
        currency="USD",
        expected_valid=False,
        injected_errors=["arithmetic_mismatch"],
        raw_text="\n".join(inv3_lines),
    )
    save_ground_truth(inv3_gt, GROUND_TRUTH_DIR / f"{inv3_id}.json")
    invoices_generated += 1

    # Invoice 4: Missing Total Amount
    inv4_id = "inv_synth_04_missing_total"
    inv4_lines = [
        "# INVOICE (INCOMPLETE)",
        "=== separator ===",
        "[B]Vendor: Metro Printing Services",
        "Invoice Number: INV-2026-104",
        "Invoice Date: 2026-10-14",
        "Payment Due Date: 2026-10-28",
        "=== separator ===",
        "Subtotal: $150.00",
        "Tax Amount: $15.00",
        "Total: ",
        "=== separator ===",
        "Note: Grand total is omitted on document.",
    ]
    inv4_pdf = SAMPLE_INVOICES_DIR / f"{inv4_id}.pdf"
    create_pdf_from_text(inv4_lines, inv4_pdf)
    inv4_gt = InvoiceGroundTruth(
        doc_id=inv4_id,
        vendor_name="Metro Printing Services",
        invoice_number="INV-2026-104",
        invoice_date="2026-10-14",
        due_date="2026-10-28",
        subtotal=150.00,
        tax_amount=15.00,
        discount_amount=0.0,
        total_amount=None,
        currency="USD",
        expected_valid=False,
        injected_errors=["missing_total_amount"],
        raw_text="\n".join(inv4_lines),
    )
    save_ground_truth(inv4_gt, GROUND_TRUTH_DIR / f"{inv4_id}.json")
    invoices_generated += 1

    # Invoice 5: Inverted Chronology (Due Date before Invoice Date)
    inv5_id = "inv_synth_05_chronology_inverted"
    inv5_lines = [
        "# TAX INVOICE",
        "=== separator ===",
        "[B]Vendor: CloudBridge Networks",
        "Invoice Number: INV-2026-199",
        "Invoice Date: 2026-11-20",
        "Payment Due Date: 2026-10-15",
        "=== separator ===",
        "Subtotal: $600.00",
        "Tax Amount: $60.00",
        "[B]Total Amount: $660.00",
        "=== separator ===",
        "Note: Due date is set before the invoice issue date.",
    ]
    inv5_pdf = SAMPLE_INVOICES_DIR / f"{inv5_id}.pdf"
    create_pdf_from_text(inv5_lines, inv5_pdf)
    inv5_gt = InvoiceGroundTruth(
        doc_id=inv5_id,
        vendor_name="CloudBridge Networks",
        invoice_number="INV-2026-199",
        invoice_date="2026-11-20",
        due_date="2026-10-15",
        subtotal=600.00,
        tax_amount=60.00,
        discount_amount=0.0,
        total_amount=660.00,
        currency="USD",
        expected_valid=False,
        injected_errors=["due_date_before_invoice_date"],
        raw_text="\n".join(inv5_lines),
    )
    save_ground_truth(inv5_gt, GROUND_TRUTH_DIR / f"{inv5_id}.json")
    invoices_generated += 1

    # Invoice 6: Non-Positive / Negative Total
    inv6_id = "inv_synth_06_non_positive_total"
    inv6_lines = [
        "# CREDIT ADJUSTMENT INVOICE",
        "=== separator ===",
        "[B]Vendor: Greenfield Energy Corp",
        "Invoice Number: INV-2026-250",
        "Invoice Date: 2026-10-15",
        "Payment Due Date: 2026-11-15",
        "=== separator ===",
        "Subtotal: -$100.00",
        "Tax Amount: $0.00",
        "[B]Total Amount: -$100.00",
        "=== separator ===",
        "Note: Non-positive total violates invoice payable integrity.",
    ]
    inv6_pdf = SAMPLE_INVOICES_DIR / f"{inv6_id}.pdf"
    create_pdf_from_text(inv6_lines, inv6_pdf)
    inv6_gt = InvoiceGroundTruth(
        doc_id=inv6_id,
        vendor_name="Greenfield Energy Corp",
        invoice_number="INV-2026-250",
        invoice_date="2026-10-15",
        due_date="2026-11-15",
        subtotal=-100.00,
        tax_amount=0.00,
        discount_amount=0.0,
        total_amount=-100.00,
        currency="USD",
        expected_valid=False,
        injected_errors=["non_positive_total_amount"],
        raw_text="\n".join(inv6_lines),
    )
    save_ground_truth(inv6_gt, GROUND_TRUTH_DIR / f"{inv6_id}.json")
    invoices_generated += 1

    # Invoice 7: Missing Vendor Name
    inv7_id = "inv_synth_07_missing_vendor"
    inv7_lines = [
        "# BILLING STATEMENT",
        "=== separator ===",
        "Address: 77 Anonymous Boulevard, Unknown City",
        "Invoice Number: INV-2026-301",
        "Invoice Date: 2026-10-16",
        "Payment Due Date: 2026-11-16",
        "=== separator ===",
        "Subtotal: $180.00",
        "Tax Amount: $18.00",
        "[B]Total Amount: $198.00",
        "=== separator ===",
        "Note: Vendor or company name is omitted entirely.",
    ]
    inv7_pdf = SAMPLE_INVOICES_DIR / f"{inv7_id}.pdf"
    create_pdf_from_text(inv7_lines, inv7_pdf)
    inv7_gt = InvoiceGroundTruth(
        doc_id=inv7_id,
        vendor_name=None,
        invoice_number="INV-2026-301",
        invoice_date="2026-10-16",
        due_date="2026-11-16",
        subtotal=180.00,
        tax_amount=18.00,
        discount_amount=0.0,
        total_amount=198.00,
        currency="USD",
        expected_valid=False,
        injected_errors=["missing_vendor_name"],
        raw_text="\n".join(inv7_lines),
    )
    save_ground_truth(inv7_gt, GROUND_TRUTH_DIR / f"{inv7_id}.json")
    invoices_generated += 1

    # -------------------------------------------------------------
    # 2. SYNTHETIC RECEIPTS (SROIE 4-FIELD COMPATIBLE)
    # -------------------------------------------------------------

    # Receipt 1: Standard Valid Receipt
    rcpt1_id = "rcpt_synth_01_valid_standard"
    rcpt1_lines = [
        "# CYNBIT COFFEE ROASTERS",
        "14 INDUSTRIAL AREA, MALVIYA NAGAR",
        "JAIPUR, RAJASTHAN 302017",
        "TEL: +91 141 555 0199",
        "=== separator ===",
        "RECEIPT NO: RCP-88321",
        "DATE: 2026-10-18 09:42",
        "CASHIER: AMIT",
        "=== separator ===",
        "1 ESPRESSO SINGLE         4.50",
        "1 OAT MILK LATTE          6.00",
        "1 CROISSANT               5.00",
        "TAX                       1.00",
        "=== separator ===",
        "[B]TOTAL AMOUNT: $16.50",
        "CASH TENDERED:   $20.00",
        "CHANGE:           $3.50",
        "=== separator ===",
        "THANK YOU FOR VISITING CYNBIT COFFEE!",
    ]
    rcpt1_pdf = SAMPLE_RECEIPTS_DIR / f"{rcpt1_id}.pdf"
    create_pdf_from_text(rcpt1_lines, rcpt1_pdf)
    rcpt1_gt = ReceiptGroundTruth(
        doc_id=rcpt1_id,
        company="CYNBIT COFFEE ROASTERS",
        date="2026-10-18",
        address="14 INDUSTRIAL AREA, MALVIYA NAGAR, JAIPUR, RAJASTHAN 302017",
        total=16.50,
        raw_text="\n".join(rcpt1_lines),
        notes="Synthetic valid receipt for SROIE 4-field baseline testing",
    )
    save_ground_truth(rcpt1_gt, GROUND_TRUTH_DIR / f"{rcpt1_id}.json")
    receipts_generated += 1

    # Receipt 2: Receipt with Missing Date
    rcpt2_id = "rcpt_synth_02_missing_date"
    rcpt2_lines = [
        "# QUICK STOP CONVENIENCE",
        "742 EVERGREEN TERRACE, SPRINGFIELD",
        "=== separator ===",
        "ITEM A                    15.00",
        "ITEM B                    12.00",
        "TAX                        1.90",
        "=== separator ===",
        "[B]TOTAL AMOUNT: $28.90",
        "=== separator ===",
        "Note: Receipt date is omitted.",
    ]
    rcpt2_pdf = SAMPLE_RECEIPTS_DIR / f"{rcpt2_id}.pdf"
    create_pdf_from_text(rcpt2_lines, rcpt2_pdf)
    rcpt2_gt = ReceiptGroundTruth(
        doc_id=rcpt2_id,
        company="QUICK STOP CONVENIENCE",
        date=None,
        address="742 EVERGREEN TERRACE, SPRINGFIELD",
        total=28.90,
        raw_text="\n".join(rcpt2_lines),
        notes="Synthetic receipt edge-case missing date",
    )
    save_ground_truth(rcpt2_gt, GROUND_TRUTH_DIR / f"{rcpt2_id}.json")
    receipts_generated += 1

    # Receipt 3: Receipt with Zero Total
    rcpt3_id = "rcpt_synth_03_zero_total"
    rcpt3_lines = [
        "# BLUE HORIZON CAFE",
        "88 OCEAN DRIVE, MIAMI, FL 33139",
        "DATE: 2026-09-30",
        "=== separator ===",
        "PROMO ITEM: COMPLIMENTARY WATER",
        "=== separator ===",
        "[B]TOTAL AMOUNT: $0.00",
        "=== separator ===",
    ]
    rcpt3_pdf = SAMPLE_RECEIPTS_DIR / f"{rcpt3_id}.pdf"
    create_pdf_from_text(rcpt3_lines, rcpt3_pdf)
    rcpt3_gt = ReceiptGroundTruth(
        doc_id=rcpt3_id,
        company="BLUE HORIZON CAFE",
        date="2026-09-30",
        address="88 OCEAN DRIVE, MIAMI, FL 33139",
        total=0.00,
        raw_text="\n".join(rcpt3_lines),
        notes="Synthetic receipt with zero total",
    )
    save_ground_truth(rcpt3_gt, GROUND_TRUTH_DIR / f"{rcpt3_id}.json")
    receipts_generated += 1

    return {
        "invoices_generated": invoices_generated,
        "receipts_generated": receipts_generated,
        "total_documents": invoices_generated + receipts_generated,
    }


if __name__ == "__main__":
    result = generate_all_synthetic_documents()
    print(f"Synthetic generation complete: {result}")
