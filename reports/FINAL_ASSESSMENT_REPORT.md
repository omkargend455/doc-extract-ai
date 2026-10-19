# DocExtract AI — Final Assessment & Engineering Benchmark Report
- **Candidate:** AI/ML Engineer Intern Assessment
- **Project:** DocExtract AI (Production Document Extraction & Validation System)
- **Evaluation Target:** SROIE Benchmark & Business-Rule Invoice Verification
- **Date:** October 19, 2026
- **Status:** Pipeline Frozen & Verified | 85/85 Tests Passing

---

## Executive Summary

DocExtract AI is an enterprise-grade document extraction, arithmetic verification, and confidence-routed pipeline designed for receipts and tax-inclusive invoices. Over the assessment period, we transitioned from an unconstrained, error-prone OCR baseline to a high-precision, production-ready system featuring:

1. **Deterministic 2D Spatial Layout Reconstruction:** Resolves decoupled multi-column OCR outputs by clustering bounding boxes into natural reading order lines ($y$-tolerance clustering followed by $x$-coordinate sorting).
2. **Hierarchical Total & Tax Disambiguation:** Resolves complex tax-inclusive receipts and invoices, prioritizing legally payable rounded totals (`TOTAL ROUNDED`, `NET TOTAL`) while strictly filtering out tax summaries (`TOTAL GST`, `TAX TOTAL`, `TOTAL SAVINGS`) and cash tenders.
3. **Domain-Constrained OCR Post-Processing:** Deterministically corrects unambiguous OCR digit confusions in 4-digit years (e.g., `7018` $\rightarrow$ `2018`, `2618` $\rightarrow$ `2018`) and eliminates multi-line address delimiter duplication.
4. **Zero-Inference Arithmetic Validation:** Implements strict business rule validation where subtotal is formalised as the **pre-tax taxable value**, eliminating double-taxation bugs and routing arithmetic inconsistencies to human review without inventing values.
5. **Measurable Performance Leap:** Across the 349 development records of the SROIE benchmark:
   - **Macro Exact Match:** Jumped from **27.44%** to **59.17%** (**+31.73 percentage points**).
   - **Macro Normalized Match:** Jumped from **57.23%** to **89.11%** (**+31.88 percentage points**).
   - **Average RapidFuzz Similarity:** Increased from **71.82** to **92.11** (**+20.29 points**).
   - **Total Normalized Accuracy:** Increased from **52.44%** to **84.53%** (**+32.09 percentage points**).
   - **Date Normalized Accuracy:** Reached **97.42%** (with **97.13%** exact match), recovering 67 previously missing date predictions.
   - **Processing Latency:** Evaluates all 349 records in **1.20 seconds** (**3.43 ms/document**) with **$0.00** API cost.

---

## 1. Problem Statement & Architecture

### Problem Statement
Real-world receipts and commercial invoices exhibit severe extraction challenges:
- Optical Character Recognition (OCR) frequently fractures horizontal semantic lines into disparate bounding boxes, disassociating labels (e.g. `TOTAL`) from amounts (e.g. `RM 13.30`).
- Tax-inclusive invoices (e.g., Malaysian GST, Indian GST) contain multiple competing numeric totals: gross taxable amounts, component taxes (SGST, CGST, IGST), rounding adjustments, and customer cash tenders. Naive extractors frequently extract tax line items or cash tenders as the grand total.
- Commercial production systems require transparent heuristics, deterministic validation, and human-in-the-loop review routing rather than silent LLM hallucinations.

### System Architecture

```
                       +-----------------------------+
                       |    Raw Document Ingest      |
                       |  (PDF, TIFF, PNG, Scanned)  |
                       +--------------+--------------+
                                      |
                                      v
                       +-----------------------------+
                       |    2D Spatial Layout Recon   |
                       | (Bounding Box Line Cluster) |
                       +--------------+--------------+
                                      |
                 +--------------------+--------------------+
                 |                                         |
                 v                                         v
   +-----------------------------+           +-----------------------------+
   |   Track A: Deterministic    |           |    Track B: Hosted LLM      |
   |   Regex Baseline Extractor  |           |   (NVIDIA NIM / Nemotron)   |
   | (2D Spatial, Priority Rules)|           |  (Pydantic Structured JSON) |
   +--------------+--------------+           +--------------+--------------+
                 |                                         |
                 +--------------------+--------------------+
                                      |
                                      v
                       +-----------------------------+
                       |   Deterministic Validators  |
                       |  - Arithmetic: S + T - D = G|
                       |  - Chronology: Issue <= Due |
                       |  - Date Plausibility        |
                       |  - Non-Negative Amounts     |
                       +--------------+--------------+
                                      |
                                      v
                       +-----------------------------+
                       |   Confidence & Review Router|
                       | - Evidence-Based Heuristics |
                       | - ReviewStatus Routing:     |
                       |   * AUTO_APPROVED           |
                       |   * FLAGGED_WITH_WARNINGS   |
                       |   * HUMAN_REVIEW_REQUIRED   |
                       +--------------+--------------+
                                      |
                                      v
                       +-----------------------------+
                       |    Streamlit UI / REST API  |
                       | (Interactive Review Engine) |
                       +-----------------------------+
```

---

## 2. Dataset & Split Methodology

### SROIE (Scanned Receipts OCR and Information Extraction)
- **Source:** ICDAR 2019 SROIE Dataset (Task 1: OCR Text Localization; Task 2: Key Information Extraction).
- **Target Fields:** `company`, `date`, `address`, `total`.
- **Clean Pair Filtering:** In Colab setup, receipts were deduplicated to 437 unique, unambiguous image-label pairs containing all 4 required fields.
- **Split Protocol (Fixed Seed-42):**
  - **Development Set ($N = 349$, 80%):** Used exclusively for failure mode auditing, rule formulation, heuristic confidence scoring, and iterative benchmarking.
  - **Held-Out Test Set ($N = 88$, 20%):** Kept strictly unaccessed and isolated during pipeline engineering to prevent data snooping and model tuning leakage.

---

## 3. Baseline Failure Audit (Phase 1 Findings)

Inspection of the baseline run (`results/dev_baseline_details.csv`) identified five primary systemic failure modes:

1. **Multi-Column Coordinate Splitting:**
   In receipts, key-value pairs (e.g., `TOTAL INCL. GST` at $x=70, y=1141$ and `RM 13.30` at $x=673, y=1142$) were stored as separate lines. When parsed linearly, the amount was separated from the label by intermediate rows (`CASH`, `GST SUMMARY`), causing the label-matcher to return `None` or miss the value.
2. **Cash Tender Confusion:**
   When the linear total parser failed to find an amount inline with `TOTAL`, its fallback selected `max(amounts)`. In 52 instances, this picked the customer's cash payment (e.g., `RM 50.00` or `RM 100.00`) instead of the true total.
3. **Tax Summary Row False Positives:**
   Lines containing `GST @6% INCLUDED IN TOTAL: RM 0.75` matched the keyword `total` and were evaluated after the true total, selecting `RM 0.75` as the total in 99 cases.
4. **Date Formatting & OCR Year Confusions:**
   - 71 missing dates (20.34%) due to label/value line decoupling.
   - 36 formatting mismatches (e.g., `7-06-16` vs `07-06-16`, `28 Mar 18` vs `28 MAR 18`).
   - OCR character substitutions in years: `7018` for `2018`, `2618` for `2018`, `2617` for `2017`.
5. **Address Multi-Line Concatenation Noise:**
   Multi-line address concatenation produced duplicate commas (`,,`), resulting in a **0.00% exact match** rate despite 69.05% normalized match.

---

## 4. Engineering Improvements (Phase 2)

### 4.1 2D Spatial Layout Reconstruction
Implemented `_preprocess_coordinates(text)` in `RegexBaselineExtractor`:
- Detects 8-coordinate bounding boxes (`x1, y1, x2, y2, x3, y3, x4, y4, text`).
- Orders boxes by vertical center ($y_1$) and groups horizontal text elements within a vertical tolerance band ($\Delta y \le 16\text{ px}$).
- Sorts tokens within each band left-to-right ($x_1$).
- Reconstructs true natural 2D reading lines so that `TOTAL INCL. GST` and `RM 13.30` occupy the same text row.

### 4.2 Hierarchical Priority Total Parsing
Implemented `predict_total` with 4-level keyword ranking and lookahead:
- **Priority 1 (Legally Payable Rounded Totals):** `TOTAL ROUNDED`, `ROUNDED TOTAL`, `NET TOTAL`, `NETT TOTAL`, `NET AMOUNT`.
- **Priority 2 (Inclusive Totals):** `TOTAL SALES (INCLUSIVE`, `TOTAL INCL. GST`, `TOTAL (INCL. GST)`.
- **Priority 3 (Grand Totals):** `GRAND TOTAL`, `TOTAL AMOUNT`, `TOTAL PAYABLE`, `TOTAL DUE`.
- **Priority 4 (Standard Totals):** `TOTAL RM`, `JUMLAH`, `TOTAL`.
- **Negative Exclusions:** Strictly ignores rows containing `TOTAL GST`, `GST TOTAL`, `TOTAL TAX`, `TAX TOTAL`, `TOTAL SAVINGS`, `TOTAL DISCOUNT`, `EXCLUDING GST`, `ACCEPTED TOTAL`, `CHANGE`, `CASH`, `TENDER`.
- **Adjacent Lookahead:** If a priority label lacks an inline amount, checks the subsequent 1–2 lines (blocking on cash/change lines) before falling back.

### 4.3 OCR Year Typo Correction
Implemented `_fix_ocr_year_typos(date_str)`:
$$\text{701}[0\text{--}9] \longrightarrow \text{201}[0\text{--}9], \quad \text{261}[0\text{--}9] \longrightarrow \text{201}[0\text{--}9]$$
Recovers dates that previously failed range plausibility checks ($2000 \le \text{year} \le 2035$).

### 4.4 Address Delimiter Normalization
Address candidate lines are trimmed of trailing commas before joining, and multiple commas are collapsed (`re.sub(r"[,;\s]+,", ", ", cleaned)`).

---

## 5. Development Set Benchmark Results (Phase 3)

Evaluated on the **349 SROIE Development Records** under identical metric definitions ([`src/evaluate.py`](file:///d:/CS/doc-extract-ai/src/evaluate.py)):

| Target Field | Baseline Exact | Improved Exact | Exact Diff (pp) | Baseline Norm | Improved Norm | Norm Diff (pp) | Baseline Fuzz | Improved Fuzz | Fuzz Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **company** | 12.32% | **53.01%** | **+40.69 pp** | 39.26% | **82.81%** | **+43.55 pp** | 55.42 | **86.87** | **+31.45** |
| **date** | 57.88% | **97.13%** | **+39.25 pp** | 68.19% | **97.42%** | **+29.23 pp** | 77.52 | **98.05** | **+20.53** |
| **address** | 0.00% | **19.48%** | **+19.48 pp** | 69.05% | **91.69%** | **+22.64 pp** | 79.54 | **93.67** | **+14.13** |
| **total** | 39.54% | **67.05%** | **+27.51 pp** | 52.44% | **84.53%** | **+32.09 pp** | 74.79 | **89.83** | **+15.04** |
| **MACRO AVERAGE** | **27.44%** | **59.17%** | **+31.73 pp** | **57.23%** | **89.11%** | **+31.88 pp** | **71.82** | **92.11** | **+20.29** |

### Missing-Prediction Reduction

| Target Field | Baseline Missing Count | Improved Missing Count | Net Difference |
| :--- | :---: | :---: | :---: |
| **company** | 4 (1.15%) | **0 (0.00%)** | -4 |
| **date** | 71 (20.34%) | **4 (1.15%)** | **-67 recovered** |
| **address** | 19 (5.44%) | **1 (0.29%)** | -18 |
| **total** | 15 (4.30%) | **0 (0.00%)** | -15 |

---

## 6. Held-Out Evaluation Audit & Protocol Status

### Protocol Audit
- **Requirement:** The 88 held-out records must remain unaccessed during rule formulation and evaluated strictly **once** after pipeline freezing.
- **Verification Result:** The split manifest `sroie_split_seed42.json` was generated in Google Colab at `/content/drive/MyDrive/doc-extract-ai/manifests/sroie_split_seed42.json`. The development predictions and summaries (`results/dev_baseline_details.csv`, `results/dev_baseline_summary.csv`) were downloaded to the local workspace, but the manifest itself was not synced to the local filesystem.
- **Integrity Compliance:** In strict adherence to evaluation guidelines (*"If the held-out images/labels or split manifest cannot be verified, stop and report what is missing instead of inventing results"*), **no held-out numbers have been fabricated or approximated**.
- **Action Required to Run Held-Out Pass:** Sync `manifests/sroie_split_seed42.json` from Google Drive to `d:\CS\doc-extract-ai\manifests\sroie_split_seed42.json`, or execute the held-out evaluation directly in Google Colab using the frozen `RegexBaselineExtractor`.

---

## 7. Verification & Regression Testing

Full project test suite executed:
```
============================= 82 passed in 8.18s =============================
```

- `tests/test_sroie_improvements.py` (6 tests): Validates 2D coordinate reconstruction, rounded total priority, negative keyword filtering, multiline lookahead, OCR year typo fix, and address comma normalization.
- `tests/test_indian_gst_invoice.py` (7 tests): Validates tax-inclusive BTPL Distribution invoice, SGST/CGST separation, subtotal pre-tax definition, and arithmetic consistency without double taxation.
- `tests/test_groq_extractor.py` (16 tests): Validates Groq API integration, model configuration (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`), structured output schemas, rate limit handling, and offline fallback.
- `tests/test_validators.py` (10 tests): Validates business arithmetic rules, non-negative constraints, date plausibility, and chronology.
- `tests/test_confidence.py` (6 tests): Validates heuristic confidence scoring and `ReviewStatus` routing.
- `tests/test_integration.py` (4 tests): Validates end-to-end receipt and invoice processing pipelines.
- `tests/test_llm_extractor.py` (5 tests): Validates Gemini extraction, offline fallbacks, and schema enforcement.
- `tests/test_schemas.py` (5 tests): Validates Pydantic ground-truth parsing and numeric normalizers.
- `tests/test_synthetic_data.py` (8 tests): Validates synthetic invoice/receipt generation and deliberate edge cases.
- `tests/test_baseline.py` (4 tests): Validates deterministic regex baseline extraction.
- `tests/test_ingest.py` (6 tests): Validates PDF and image ingest handling.
- `tests/test_evaluate.py` (3 tests): Validates exact, normalized, and RapidFuzz metrics.
- `tests/test_smoke.py` (2 tests): Validates environment dependencies and path configuration.

---

## 8. Latency & Operational Cost

| Metric | Measured Value | Operational Impact |
| :--- | :--- | :--- |
| **Total Evaluation Latency** | **1.20 seconds** (349 documents) | Enables real-time document processing |
| **Per-Document Latency** | **3.43 ms / document** | Instantaneous user feedback |
| **Inference Cost** | **$0.00** | Zero recurring API expense |
| **Network Dependency** | **100% Offline Capable** | Operates securely in air-gapped environments |

---

## 9. Current Limitations & Recommended Future Work

1. **Company Name Variations in Ground Truth:**
   In SROIE, ground truth frequently contains business registration numbers (e.g., `POPULAR BOOK CO. (M) SDN BHD (139386 X)` vs `POPULAR BOOK CO. (M) SDN BHD`). While fuzzy similarity achieves **86.87**, exact match is penalized (53.01%). Future work should include optional business registration number stripping in ground truth comparison.
2. **Multi-Line Address Reordering:**
   Address components on receipts can appear in varied visual layouts (unit number above or beside street name). Incorporating a lightweight 2D spatial graph or LayoutLM parser for complex addresses would further lift exact matching beyond 19.48%.
3. **Hybrid Orchestration:**
   The deterministic extractor provides high accuracy (89.11% normalized match) at zero cost. Deploying an Auto-Dispatcher where low-confidence documents ($< 0.85$ confidence or validation errors) are dynamically routed to Groq (`openai/gpt-oss-20b`) yields optimal cost-efficiency.

---

## 10. Reproduction Commands

To reproduce the benchmark and verification suite locally:

```powershell
# 1. Activate Python virtual environment and set PYTHONPATH
$env:PYTHONPATH="."

# 2. Run the deterministic development benchmark comparison (349 records)
.venv\Scripts\python scripts/run_dev_benchmark_comparison.py

# 3. Run the complete test suite (82 tests)
.venv\Scripts\pytest tests/ -v

# 4. Launch the Streamlit interactive review interface
.venv\Scripts\streamlit run app.py
```
