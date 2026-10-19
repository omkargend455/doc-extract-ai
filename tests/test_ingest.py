"""Unit tests for document ingestion engine (PyMuPDF digital extraction & image handling)."""

import io
from pathlib import Path
from unittest.mock import patch
import pytest
from PIL import Image

from src.config import SAMPLE_INVOICES_DIR, IS_TESSERACT_AVAILABLE
from src.ingest import (
    extract_text_from_pdf,
    extract_text_from_image,
    ingest_document,
    TesseractNotFoundError,
    IngestionResult,
)


def test_pdf_extraction_from_file():
    """Verify PyMuPDF digital text extraction on a synthetic digital invoice PDF."""
    pdf_path = SAMPLE_INVOICES_DIR / "inv_synth_01_valid_standard.pdf"
    assert pdf_path.exists(), "Sample invoice PDF must exist"

    result = extract_text_from_pdf(pdf_path)

    assert isinstance(result, IngestionResult)
    assert result.source_type == "pdf_digital"
    assert result.page_count == 1
    assert result.character_count > 100
    assert not result.is_empty
    assert not result.ocr_applied
    assert "Apex Office Supplies Ltd" in result.text
    assert "INV-2026-001" in result.text
    assert "Total Amount Due: $220.00" in result.text


def test_pdf_extraction_from_bytes_and_buffer():
    """Verify PDF extraction when given raw bytes or io.BytesIO stream."""
    pdf_path = SAMPLE_INVOICES_DIR / "inv_synth_01_valid_standard.pdf"
    raw_bytes = pdf_path.read_bytes()

    # From raw bytes
    res_bytes = extract_text_from_pdf(raw_bytes)
    assert not res_bytes.is_empty
    assert "Apex Office Supplies Ltd" in res_bytes.text

    # From file-like BytesIO stream
    stream = io.BytesIO(raw_bytes)
    res_stream = extract_text_from_pdf(stream)
    assert not res_stream.is_empty
    assert "Apex Office Supplies Ltd" in res_stream.text


def test_pdf_extraction_missing_file():
    """Verify FileNotFoundError is raised when file does not exist."""
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf(Path("missing_doc.pdf"))


def test_image_extraction_when_tesseract_unavailable(tmp_path):
    """Verify graceful handling when Tesseract is not installed on the system."""
    dummy_image = tmp_path / "test_scan.png"
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    img.save(dummy_image)

    # Force IS_TESSERACT_AVAILABLE to False for deterministic test
    with patch("src.ingest.IS_TESSERACT_AVAILABLE", False):
        # 1. Non-raising mode (default): returns empty result with actionable warning
        result = extract_text_from_image(dummy_image, raise_if_missing_tesseract=False)
        assert result.is_empty
        assert not result.ocr_applied
        assert result.warning is not None
        assert "Tesseract OCR is not installed" in result.warning
        assert "https://github.com/UB-Mannheim/tesseract/wiki" in result.warning

        # 2. Raising mode: raises TesseractNotFoundError with actionable message
        with pytest.raises(TesseractNotFoundError) as exc_info:
            extract_text_from_image(dummy_image, raise_if_missing_tesseract=True)
        assert "Tesseract OCR is not installed" in str(exc_info.value)


def test_image_extraction_when_tesseract_mocked(tmp_path):
    """Verify OCR branch logic when Tesseract is present."""
    dummy_image = tmp_path / "mock_scan.png"
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    img.save(dummy_image)

    with patch("src.ingest.IS_TESSERACT_AVAILABLE", True):
        with patch("src.ingest.pytesseract.image_to_string", return_value="STORE RECEIPT\nTOTAL $15.00"):
            result = extract_text_from_image(dummy_image)
            assert result.ocr_applied
            assert not result.is_empty
            assert "STORE RECEIPT" in result.text
            assert result.character_count > 0


def test_unified_ingest_dispatcher(tmp_path):
    """Verify routing between PDF, image, and raw text formats."""
    pdf_path = SAMPLE_INVOICES_DIR / "inv_synth_01_valid_standard.pdf"
    res_pdf = ingest_document(pdf_path)
    assert res_pdf.source_type == "pdf_digital"

    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("Hello plain text document", encoding="utf-8")
    res_txt = ingest_document(txt_path)
    assert res_txt.source_type == "raw_text"
    assert "Hello plain text document" in res_txt.text
