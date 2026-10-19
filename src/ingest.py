"""Document ingestion engine for digital PDFs and scanned images."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Optional
import pymupdf
from PIL import Image
import pytesseract

from src.config import IS_TESSERACT_AVAILABLE, TESSERACT_PATH

TESSERACT_GUIDANCE_MESSAGE = (
    "Tesseract OCR is not installed or not found on the system PATH. "
    "Image OCR requires Tesseract. "
    "To enable image processing on Windows:\n"
    "  1. Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki\n"
    "  2. Add the installation folder (e.g., C:\\Program Files\\Tesseract-OCR) to your system PATH.\n"
    "  3. Restart your terminal or virtual environment.\n"
    "Note: Digital PDFs with native text layers do not require Tesseract and can be processed immediately."
)


class TesseractNotFoundError(RuntimeError):
    """Raised when OCR is requested on an image but Tesseract is unavailable on the system."""

    def __init__(self, message: str = TESSERACT_GUIDANCE_MESSAGE):
        super().__init__(message)


@dataclass
class IngestionResult:
    """Standardized representation of ingested document text and extraction metadata."""

    text: str
    source_type: str  # "pdf_digital", "pdf_scanned", "image", "raw_text"
    page_count: int = 1
    character_count: int = 0
    is_empty: bool = True
    ocr_applied: bool = False
    warning: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    image_bytes: Optional[bytes] = None


def extract_text_from_pdf(pdf_source: str | Path | bytes | BinaryIO) -> IngestionResult:
    """Extract digital text from a PDF document using PyMuPDF (fitz).

    Supports file paths, raw bytes, or file-like buffer streams (e.g. Streamlit file uploads).
    """
    doc: pymupdf.Document
    if isinstance(pdf_source, (str, Path)):
        path = Path(pdf_source)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        doc = pymupdf.open(str(path))
        filename = path.name
    elif isinstance(pdf_source, bytes):
        doc = pymupdf.open(stream=pdf_source, filetype="pdf")
        filename = "in_memory.pdf"
    elif hasattr(pdf_source, "read"):
        content = pdf_source.read()
        if hasattr(pdf_source, "seek"):
            pdf_source.seek(0)
        doc = pymupdf.open(stream=content, filetype="pdf")
        filename = getattr(pdf_source, "name", "uploaded.pdf")
    else:
        raise ValueError(f"Unsupported PDF source type: {type(pdf_source)}")

    try:
        pages_text: list[str] = []
        page_count = len(doc)
        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text("text") or ""
            pages_text.append(text)

        full_text = "\n".join(pages_text).strip()
        char_count = len(full_text)
        is_empty = char_count == 0

        warning = None
        source_type = "pdf_digital"

        # Heuristic: If PDF has multiple pages but virtually no text, it is likely a scanned image PDF
        if page_count > 0 and char_count < (15 * page_count):
            source_type = "pdf_scanned"
            warning = (
                "PDF contains minimal or no extractable digital text. "
                "The file appears to be a scanned document or image-only PDF."
            )

        # Render first page to jpeg for optional visual/multimodal model inspection
        image_bytes = None
        try:
            if page_count > 0:
                pix = doc[0].get_pixmap(dpi=150)
                image_bytes = pix.tobytes("jpeg")
        except Exception:
            image_bytes = None

        return IngestionResult(
            text=full_text,
            source_type=source_type,
            page_count=page_count,
            character_count=char_count,
            is_empty=is_empty,
            ocr_applied=False,
            warning=warning,
            metadata={
                "filename": filename,
                "pdf_metadata": doc.metadata,
                "page_count": page_count,
            },
            image_bytes=image_bytes,
        )
    finally:
        doc.close()


def extract_text_from_image(
    image_source: str | Path | bytes | BinaryIO | Image.Image,
    raise_if_missing_tesseract: bool = False,
) -> IngestionResult:
    """Extract text from an image using Tesseract OCR.

    If Tesseract is not found on the host system, returns an empty IngestionResult
    with actionable user guidance unless raise_if_missing_tesseract is True.
    """
    if not IS_TESSERACT_AVAILABLE:
        if raise_if_missing_tesseract:
            raise TesseractNotFoundError(TESSERACT_GUIDANCE_MESSAGE)
        return IngestionResult(
            text="",
            source_type="image",
            page_count=1,
            character_count=0,
            is_empty=True,
            ocr_applied=False,
            warning=TESSERACT_GUIDANCE_MESSAGE,
            metadata={"tesseract_installed": False},
        )

    # Load PIL image
    img: Image.Image
    filename = "image"
    if isinstance(image_source, Image.Image):
        img = image_source
    elif isinstance(image_source, (str, Path)):
        path = Path(image_source)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        img = Image.open(path)
        filename = path.name
    elif isinstance(image_source, bytes):
        img = Image.open(io.BytesIO(image_source))
    elif hasattr(image_source, "read"):
        content = image_source.read()
        if hasattr(image_source, "seek"):
            image_source.seek(0)
        img = Image.open(io.BytesIO(content))
        filename = getattr(image_source, "name", "uploaded_image")
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")

    # Run Tesseract OCR
    ocr_text = pytesseract.image_to_string(img) or ""
    cleaned_text = ocr_text.strip()
    char_count = len(cleaned_text)

    # Convert image to jpeg bytes for optional visual/multimodal model inspection
    image_bytes = None
    try:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        image_bytes = buf.getvalue()
    except Exception:
        image_bytes = None

    return IngestionResult(
        text=cleaned_text,
        source_type="image",
        page_count=1,
        character_count=char_count,
        is_empty=(char_count == 0),
        ocr_applied=True,
        warning=None if char_count > 0 else "OCR completed but no legible text was detected in the image.",
        metadata={
            "filename": filename,
            "image_size": img.size,
            "image_mode": img.mode,
            "tesseract_path": TESSERACT_PATH,
        },
        image_bytes=image_bytes,
    )


def ingest_document(
    source: str | Path | bytes | BinaryIO,
    filename: Optional[str] = None,
    raise_if_missing_ocr: bool = False,
) -> IngestionResult:
    """Unified document ingestion dispatcher supporting digital PDFs, images, and text.

    Inspects file extension or filename to select the optimal ingestion pipeline.
    """
    # Determine filename
    target_name = ""
    if filename:
        target_name = filename.lower()
    elif isinstance(source, (str, Path)):
        target_name = Path(source).name.lower()
    elif hasattr(source, "name"):
        target_name = str(getattr(source, "name", "")).lower()

    if target_name.endswith(".pdf"):
        return extract_text_from_pdf(source)

    if any(target_name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]):
        return extract_text_from_image(source, raise_if_missing_tesseract=raise_if_missing_ocr)

    if target_name.endswith(".txt"):
        if isinstance(source, (str, Path)):
            text = Path(source).read_text(encoding="utf-8")
        elif isinstance(source, bytes):
            text = source.decode("utf-8", errors="replace")
        elif hasattr(source, "read"):
            raw = source.read()
            text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        else:
            text = str(source)
        cleaned = text.strip()
        return IngestionResult(
            text=cleaned,
            source_type="raw_text",
            page_count=1,
            character_count=len(cleaned),
            is_empty=(len(cleaned) == 0),
            ocr_applied=False,
            metadata={"filename": target_name},
        )

    # Fallback attempt: Try PDF first, if error try Image
    try:
        return extract_text_from_pdf(source)
    except Exception:
        return extract_text_from_image(source, raise_if_missing_tesseract=raise_if_missing_ocr)
