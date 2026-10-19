\# DocExtract AI



\## Project Description



DocExtract AI is an AI-powered document information extraction and validation system designed to convert unstructured documents, such as receipts and invoices, into structured, machine-readable data.



The project aims to combine traditional rule-based extraction techniques with Large Language Model (LLM)-based extraction to identify important fields, including document dates, vendor details, invoice numbers, and total amounts. Extracted information will be validated against predefined schemas and business rules to improve consistency and identify potentially incorrect or incomplete results.



The system will also explore confidence and review indicators to flag documents that may require human verification. Its performance will be evaluated using labelled datasets and synthetic test documents, comparing extraction accuracy and validation outcomes across different approaches.



A lightweight Streamlit interface will allow users to upload documents, inspect extracted information, review validation results, and understand when manual verification may be needed.



\## Project Objectives



\- Extract structured information from receipts and invoices.

\- Compare rule-based extraction with LLM-based structured extraction.

\- Validate extracted fields using predefined schemas and business rules.

\- Identify incomplete, inconsistent, or potentially incorrect extractions.

\- Evaluate extraction performance using measurable metrics.

\- Provide an interactive interface for document processing and result review.



\## Technology Stack



\- \*\*Language:\*\* Python

\- \*\*Interface:\*\* Streamlit

\- \*\*Document Processing:\*\* PyMuPDF and OCR tools

\- \*\*Data Validation:\*\* Pydantic

\- \*\*AI-Based Extraction:\*\* Gemini API

\- \*\*Evaluation:\*\* Pandas and RapidFuzz

\- \*\*Testing:\*\* pytest



\*Note: The technology stack and objectives describe the planned implementation and may evolve during development.\*



\## Project Status

Currently in the initial setup and development phase (Milestone 1.1: Foundation & Smoke Test verified).

## Getting Started & Local Setup

### 1. Prerequisites
- **Python:** Version 3.11 to 3.13 recommended.
- **Git:** Standard git version control.
- **Tesseract OCR (Optional for scanned image OCR):**
  - Windows: Install via [UB-Mannheim Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki) and add to system PATH.
  - Linux / Streamlit Cloud: Configured automatically via `packages.txt`.
  - *Note:* Digital PDFs are parsed natively via PyMuPDF without requiring Tesseract.

### 2. Environment Setup

Clone the repository and navigate into the root directory:
```bash
git clone https://github.com/<your-username>/cynbit-doc-extraction.git
cd doc-extract-ai
```

Create and activate a virtual environment:
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install the pinned dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration (Optional for Milestone 1.1)

Copy the sample environment file:
```bash
cp .env.example .env
```
Add your Google Gemini API key if testing intelligent LLM extraction:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(If unset, the application automatically runs in zero-cost offline mode utilizing the deterministic baseline).*

### 4. Running the Application

Launch the Streamlit interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Running Tests

Execute the automated test suite with pytest:
```bash
pytest tests/ -v
```

