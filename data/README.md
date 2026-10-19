# DocExtract AI - Dataset Governance & Acquisition Guide

This directory manages document datasets and benchmark annotations used for evaluating document extraction accuracy and validating business arithmetic rules.

---

## Directory Layout

```text
data/
├── sample_receipts/        # Scanned receipt PDFs, images & text files
├── sample_invoices/        # Synthetic digital invoice PDFs for arithmetic verification
├── ground_truth/           # Standardized JSON annotations for objective evaluation
├── generate_synthetic.py   # Deterministic synthetic invoice/receipt testbed generator
├── raw/                    # (Git-ignored) Unprocessed external downloads
├── processed/              # (Git-ignored) Intermediate processed artifacts
└── README.md               # Data governance and acquisition documentation
```

---

## Data Policy & Lawful Dataset Usage

### 1. SROIE Dataset (ICDAR 2019 Robust Reading Challenge)

The primary real-world benchmark used for receipt extraction evaluation is derived from the **ICDAR 2019 Scanned Receipts OCR and Information Extraction (SROIE)** competition.

- **Official Challenge Host:** [ICDAR 2019 Robust Reading Competition](https://rrc.cvc.uab.es/?ch=13)
- **Challenge Tasks:**
  - Task 1: Scanned Receipt Text Localisation
  - Task 2: Scanned Receipt OCR
  - Task 3: Key Information Extraction (KIE)
- **Target Fields in Task 3:**
  1. `company`: Store or vendor name
  2. `date`: Date of transaction (standardized or original string)
  3. `address`: Store street address
  4. `total`: Total currency amount paid
- **Licensing & Redistribution Policy:**
  - The SROIE dataset is hosted and copyrighted by its respective challenge organizers and research institutions under academic research terms.
  - To strictly respect dataset licensing, terms of service, and distribution conditions, third-party SROIE images are **not redistributed** directly within this git repository.

#### How to Obtain and Use SROIE Lawfully:
1. Navigate to the official competition portal at [https://rrc.cvc.uab.es/?ch=13](https://rrc.cvc.uab.es/?ch=13).
2. Create an academic/research account and accept the dataset terms of use.
3. Download the Task 3 (Key Information Extraction) test dataset.
4. Extract the images into `data/sample_receipts/`.
5. Convert or copy the provided ground-truth `.txt` or `.json` annotations into `data/ground_truth/<doc_id>.json` following the schema below.

#### Standardized Receipt Ground-Truth Format (`ReceiptGroundTruth`):
```json
{
  "doc_id": "receipt_001",
  "doc_type": "receipt",
  "company": "BOOKSTORE CO.",
  "date": "2026-03-15",
  "address": "123 MAIN ST, CITY, STATE",
  "total": 45.50,
  "notes": "SROIE Task 3 annotation"
}
```

---

## 2. Synthetic Testbed (Invoice & Receipt Generator)

To enable 100% reproducible testing of deterministic business rules without external licensing restrictions, the repository includes a programmatic dataset generator:

```bash
python data/generate_synthetic.py
```

This generates:
- **7 Synthetic Invoices** in `data/sample_invoices/` (real digital PDFs with native text layers generated via PyMuPDF).
- **3 Synthetic Receipts** in `data/sample_receipts/` (PDF receipts with SROIE-compatible 4-field ground truth).
- **10 Paired JSON Ground-Truth Files** in `data/ground_truth/` matching `InvoiceGroundTruth` and `ReceiptGroundTruth` schemas.

### Deliberate Edge Cases Covered in Synthetic Suite:
1. `inv_synth_01_valid_standard`: Standard valid tax invoice (Subtotal \$200 + Tax \$20 = Total \$220).
2. `inv_synth_02_valid_discount`: Multi-item invoice with discount (Subtotal \$450 + Tax \$36 - Discount \$50 = Total \$436).
3. `inv_synth_03_arithmetic_mismatch`: Subtotal (\$300) + Tax (\$30) != Total printed as \$375 (injected calculation error).
4. `inv_synth_04_missing_total`: Grand total line is omitted.
5. `inv_synth_05_chronology_inverted`: Payment due date is chronologically before invoice issue date.
6. `inv_synth_06_non_positive_total`: Grand total is negative (-\$100.00).
7. `inv_synth_07_missing_vendor`: Vendor/company name is completely omitted.
8. `rcpt_synth_01_valid_standard`: SROIE-compliant receipt with all 4 target fields populated.
9. `rcpt_synth_02_missing_date`: Receipt with missing date.
10. `rcpt_synth_03_zero_total`: Receipt with zero total amount (\$0.00).
