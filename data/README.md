# DocExtract AI - Dataset Directory & Data Policy

This directory manages document datasets and benchmark annotations used for evaluating document extraction accuracy.

## Directory Structure

```text
data/
├── sample_receipts/        # Scanned receipt images & text for extraction testing
├── sample_invoices/        # Synthetic & sample invoices for arithmetic verification
├── ground_truth/           # Standardized JSON annotations for objective evaluation
├── raw/                    # (Git-ignored) Unprocessed external downloads
├── processed/              # (Git-ignored) Intermediate processed artifacts
└── README.md               # Data governance and acquisition documentation
```

## Dataset Information & Acquisition

### 1. SROIE Dataset (ICDAR 2019 Robust Reading Challenge)
The primary real-world benchmark used for receipt extraction evaluation is derived from the **ICDAR 2019 Scanned Receipts OCR and Information Extraction (SROIE)** challenge.

- **Official Challenge Host:** [ICDAR 2019 Robust Reading Competition](https://rrc.cvc.uab.es/?ch=13)
- **Dataset Contents:** Scanned receipt images covering 4 target entities:
  - `company` (Vendor Name)
  - `date` (Receipt Issue Date)
  - `address` (Store Address)
  - `total` (Total Amount Paid)
- **Licensing & Redistribution Policy:**
  - Raw SROIE dataset images and challenge annotations are governed by the ICDAR competition terms and research licensing.
  - To respect copyright and distribution conditions, full external raw datasets are **not redistributed** directly within this source code repository.
  - Researchers and evaluators can register and download the dataset directly from the official ICDAR 2019 competition portal or authorized research archives.

### 2. Synthetic Invoices Testbed
To test deterministic arithmetic rules ($\text{Subtotal} + \text{Tax} - \text{Discount} == \text{Total}$) and chronological consistency without external copyright encumbrances:
- A programmatic invoice generator will be provided in Milestone 1.2 (`data/generate_synthetic.py`).
- Synthetic invoices contain controlled ground truths and deliberate edge cases (arithmetic discrepancies, inverted dates, missing totals).
