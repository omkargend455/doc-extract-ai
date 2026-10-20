# DocExtract AI

> **Document Extraction, Deterministic Business Validation & Human-in-the-Loop Triage**  
> *Developed for the Cynbit AI/ML Engineer Intern Technical Assessment.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://doc-extract-ai.streamlit.app/)
[![Tests](https://img.shields.io/badge/tests-96%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](requirements.txt)
[![Architecture](https://img.shields.io/badge/architecture-hybrid%20LLM%20%2B%20regex-orange.svg)](#system-architecture)

**Live Deployed Application:** [https://doc-extract-ai.streamlit.app/](https://doc-extract-ai.streamlit.app/)

---

## 📌 Overview

**DocExtract AI** is a document extraction and validation pipeline designed to parse unstructured invoices and receipts into validated structured records. 

Rather than relying solely on raw generative outputs, DocExtract AI pairs hosted multimodal foundation models with a fast deterministic baseline, enforces strict business rule validation (such as accounting cross-consistency and chronological checks), and calculates evidence-grounded confidence scores to route documents through a 3-way review decider.

---

## 🚀 Key Capabilities

- **Configurable Extraction Modes:**
  - **`Auto` Mode (Default):** Queries hosted NVIDIA NIM models when an API key is available. If unconfigured or if an API failure occurs (e.g., authentication failure, rate limit, timeout, or malformed JSON), it automatically falls back to the deterministic regex engine with a sanitized diagnostic message.
  - **`NVIDIA NIM` Mode:** Explicitly targets hosted foundation models:
    - `nvidia/nemotron-3.5-lightning-30b-a3b` (text-based extraction)
    - `nvidia/nemotron-3.5-super-vl-120b-a12b` (multimodal vision extraction)
    - If the API call fails, the exception is caught, masked to prevent credential exposure, and handled with a fallback response.
  - **`Regex Baseline` Mode:** Runs 100% locally and offline without external dependencies or API keys.
- **Layout-Aware Ingestion:**
  - Fast native PDF text extraction via PyMuPDF.
  - Spatial 2D token sorting (`(round(y, -1), x)`) to reconstruct line reading order on complex multi-column layouts.
  - Tesseract OCR fallback for scanned images and degraded mobile captures when a digital text layer is absent.
- **Pydantic v2 Schema Normalization:**
  - Standardizes extracted records into `InvoiceGroundTruth` and `ReceiptGroundTruth` schemas.
  - Preserves explicit zero taxes (`tax_amount = 0.0`) when supported by source document evidence, avoiding false positives while keeping unmentioned jurisdictional taxes (SGST, CGST, IGST) as `null`.
- **Deterministic Business Rule Validation:**
  - **Invoice Arithmetic Check:** Evaluates `|Subtotal + Tax - Discount - Total| <= 0.05` (with tax supporting general tax or itemized `SGST + CGST + IGST`). Evaluated only when both subtotal and total are present; does not fabricate unextracted values.
  - **Receipt Validation:** Evaluates presence of mandatory fields (`company`, `date`, `total`), positive amount constraints, and date validity (arithmetic cross-checks are cleanly skipped if subtotal is absent).
  - **Date Plausibility & Chronology:** Validates that dates parse into ISO format (`YYYY-MM-DD`) within plausible operational bounds (2000–2035) and enforces `Invoice Date <= Due Date`.
  - **Monetary Constraints:** Ensures final totals are strictly positive (`> 0.0`).
- **3-Way Human-in-the-Loop Triage:**
  - `AUTO_APPROVED`: Composite confidence $\ge 0.85$ with **0 validation errors** and **0 validation warnings**.
  - `FLAGGED_WITH_WARNINGS`: Moderate confidence ($0.70 \le \text{score} < 0.85$) or non-critical warnings present (0 hard errors).
  - `HUMAN_REVIEW_REQUIRED`: Low confidence ($< 0.70$) or any hard validation error (e.g., arithmetic mismatch, negative total, inverted dates).

---

## 🏗 System Architecture

```text
       ┌────────────────────────┐
       │ Ingestion (PyMuPDF)    │
       │ Text / OCR / Visual    │
       └───────────┬────────────┘
                   │
           [Extraction Mode]
         /         │        \
      Auto      NVIDIA    Baseline
       │           │         │
[Key configured?]  │         │
  YES / NO         │         │
  /      \         │         │
NIM     Regex     NIM      Regex
  │                │         │
(NIM failure?)     │         │
  └──────► Regex ◄─┘         │
             │               │
             └───────┬───────┘
                     │
                     ▼
       ┌────────────────────────────┐
       │  Pydantic v2 Normalization │
       └─────────────┬──────────────┘
                     │
                     ▼
       ┌────────────────────────────┐
       │  Deterministic Validators  │
       │  - Invoices: Balance check │
       │  - Dates: Plausibility     │
       │  - Zero-Tax Preservation   │
       └─────────────┬──────────────┘
                     │
                     ▼
       ┌────────────────────────────┐
       │ Composite Confidence (0-1) │
       └─────────────┬──────────────┘
                     │
                     ▼
       ┌────────────────────────────┐
       │  Review Decider & Routing  │
       │  - AUTO_APPROVED           │
       │  - FLAGGED_WITH_WARNINGS   │
       │  - HUMAN_REVIEW_REQUIRED   │
       └────────────────────────────┘
```

---

## 📊 Evaluation & Empirical Benchmarks

### 1. ICDAR 2019 SROIE Benchmark (349 Development Records)
Tested across real-world receipts from the ICDAR 2019 SROIE Task 2 development set:

| Field | Baseline Regex Match | Improved Hybrid Pipeline | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **Total Amount** | 57.88% | **88.83%** | **+30.95%** |
| **Date** | 64.76% | **93.12%** | **+28.36%** |
| **Address** | 70.49% | **91.69%** | **+21.20%** |
| **Company** | 35.82% | **82.81%** | **+46.99%** |
| **Macro Average** | **57.23%** | **89.11%** | **+31.88%** |

*Note on Latency:* During development benchmarking, the local deterministic regex extractor processed records in an average of **3.43 ms per document**. Hosted multimodal LLM/VLM inference involves network roundtrips and remote computation, typically requiring **2 to 5 seconds per document** depending on network conditions.

### 2. Synthetic Edge-Case Suites
- **7 Synthetic Invoices:** Verified valid invoices, discount applications, arithmetic mismatches, missing totals, inverted due dates, non-positive totals, and missing vendors.
- **3 Synthetic Receipts:** Verified standard receipts, missing dates, and zero totals.

### 3. Automated Test Suite
- **96 passing unit, integration, and regression tests** validating schema parsing, arithmetic tolerances, zero-tax preservation, and triage routing.

---

## 📁 Repository Structure

```text
doc-extract-ai/
├── app.py                      # Streamlit interactive user interface
├── requirements.txt            # Python dependencies
├── packages.txt                # Linux OS dependencies (Tesseract OCR for Cloud)
├── src/
│   ├── config.py               # Runtime settings, credentials & model definitions
│   ├── schemas.py              # Pydantic v2 schemas (Invoice & Receipt models)
│   ├── ingest.py               # PyMuPDF ingestion & 2D spatial text sorting
│   ├── baseline_regex.py       # Deterministic regex baseline extractor
│   ├── nvidia_extractor.py     # Hosted NVIDIA NIM Nemotron LLM/VLM extractor
│   ├── extractor_dispatcher.py # Provider mode dispatching & fallback routing
│   ├── validators.py           # Arithmetic, date, and business rule validators
│   ├── confidence.py           # Field-level & composite confidence scoring
│   └── evaluate.py             # SROIE benchmark evaluation harness (RapidFuzz)
├── data/
│   ├── sample_invoices/        # 7 synthetic edge-case invoice PDFs
│   ├── sample_receipts/        # 3 synthetic edge-case receipt PDFs
│   └── sroie_manifest_dev.json # 349-record SROIE development manifest
├── tests/                      # 96 automated pytest unit and regression tests
├── reports/                    # Assessment Companion Reports & PDFs
│   ├── AI_OPPORTUNITY_REPORT.md    # Business alignment, 12 AI opportunities & matrix
│   ├── TECHNICAL_APPROACH.md       # Architecture, validation math, limitations & RAG
│   └── FINAL_ASSESSMENT_REPORT.md  # Engineering audit, failure analysis & metrics
└── results/                    # SROIE benchmark summary CSVs
```

---

## 📄 Assessment Companion Reports

Companion reports detailing business and engineering analyses are located in [`reports/`](reports/):

1. **[AI Opportunity Report](reports/AI_OPPORTUNITY_REPORT.md)** ([PDF](reports/AI_OPPORTUNITY_REPORT.pdf)):
   - Analysis of Cynbit's business model and SaaS portfolio (*TenantEase*, *Cynbit Studio*, *Revio*).
   - Evaluation of 12 candidate AI opportunities scored across 7 weighted criteria.
   - 4-phase implementation roadmap and strategic alignment.
2. **[Technical Approach Report](reports/TECHNICAL_APPROACH.md)** ([PDF](reports/TECHNICAL_APPROACH.pdf)):
   - Detailed pipeline architecture, 2D spatial layout sorting, and validation math.
   - Known limitations (OCR skew, complex multi-line addresses, network latency).
   - Proposed future enhancement: **Conversational Expense Intelligence Using RAG** (deterministic SQL aggregations + vector voucher retrieval).
3. **[Final Assessment Report](reports/FINAL_ASSESSMENT_REPORT.md)**:
   - Root-cause audit of baseline regex failures and SROIE development benchmarks.

---

## 🛠 Local Setup & Running

### 1. Prerequisites
- Python 3.11 or 3.12 recommended.
- *(Optional)* [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) for scanned image OCR (digital PDFs are parsed natively via PyMuPDF).

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Omkar-Kharmate/doc-extract-ai.git
cd doc-extract-ai

# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. API Configuration (Optional)
Configure `.streamlit/secrets.toml` or create a `.env` file:
```env
NVIDIA_API_KEY=nvapi-your-key-here
```
*(If no API key is provided, the application runs offline using the deterministic regex baseline).*

### 4. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Run the Automated Tests
```bash
pytest tests/ -v
```

### 6. Run the Benchmark Harness
```bash
python -m src.evaluate --dataset dev --method hybrid
```

---

## 🔮 Future Roadmap: Conversational Expense Intelligence (RAG)

As documented in Section 7.3 of the Technical Approach Report, a proposed future extension is extending the pipeline into a **Conversational Expense Intelligence Engine**:
- **Deterministic SQL Calculations:** Exact financial totals, category summaries, and month-over-month comparisons are executed strictly via database queries.
- **RAG Document Retrieval:** Dense vector retrieval indexes document text and OCR tokens to fetch supporting vouchers, providing grounded explanations with authenticated citations.
- **Multi-Tenant Controls:** Tenant-level data isolation to ensure role-based access control across multi-client platforms like Cynbit's *TenantEase*.

---

## ⚖️ Attribution

Developed by **Omkar Kharmate** for the **Cynbit Technologies AI/ML Engineer Intern Technical Assessment**.
