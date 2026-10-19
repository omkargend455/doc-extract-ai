"""Data schemas and ground-truth models for DocExtract AI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class ReceiptGroundTruth(BaseModel):
    """Standardized ground-truth annotations for receipt documents (SROIE-compatible)."""

    doc_id: str = Field(description="Unique document identifier")
    doc_type: str = Field(default="receipt", description="Document type identifier")
    company: Optional[str] = Field(default=None, description="Issuing store or company name")
    date: Optional[str] = Field(default=None, description="Document date in YYYY-MM-DD or original format")
    address: Optional[str] = Field(default=None, description="Full store street address")
    total: Optional[float] = Field(default=None, description="Final receipt total amount paid")
    raw_text: Optional[str] = Field(default=None, description="Full OCR or digital text content")
    notes: Optional[str] = Field(default=None, description="Annotation metadata or dataset source")

    @field_validator("total", mode="before")
    @classmethod
    def normalize_total(cls, v: Any) -> Optional[float]:
        """Normalize numeric or string amounts into float."""
        if v is None or v == "":
            return None
        if isinstance(v, (int, float)):
            return float(v)
        # Handle string totals with currency symbols, spaces, and commas
        cleaned = str(v).replace("$", "").replace("RM", "").replace("₹", "").replace("€", "").replace("£", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None


class InvoiceGroundTruth(BaseModel):
    """Standardized ground-truth annotations for invoice documents."""

    doc_id: str = Field(description="Unique document identifier")
    doc_type: str = Field(default="invoice", description="Document type identifier")
    vendor_name: Optional[str] = Field(default=None, description="Issuing company or vendor name")
    invoice_number: Optional[str] = Field(default=None, description="Unique invoice or bill identifier")
    invoice_date: Optional[str] = Field(default=None, description="Issue date in YYYY-MM-DD format")
    due_date: Optional[str] = Field(default=None, description="Payment due date in YYYY-MM-DD format")
    subtotal: Optional[float] = Field(default=None, description="Subtotal before taxes and discounts")
    tax_amount: Optional[float] = Field(default=None, description="Tax or VAT amount")
    discount_amount: Optional[float] = Field(default=0.0, description="Applied discount amount")
    total_amount: Optional[float] = Field(default=None, description="Final grand total payable")
    currency: Optional[str] = Field(default="USD", description="Currency code (USD, EUR, INR, MYR)")
    items: list[dict[str, Any]] = Field(default_factory=list, description="Line item details")
    expected_valid: bool = Field(default=True, description="Whether this document passes all business arithmetic/chronology rules")
    injected_errors: list[str] = Field(default_factory=list, description="Deliberate error flags for edge-case testing")
    raw_text: Optional[str] = Field(default=None, description="Raw text of the invoice")


def load_ground_truth(file_path: Path | str) -> ReceiptGroundTruth | InvoiceGroundTruth:
    """Load and parse a ground-truth JSON file into the appropriate schema."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc_type = data.get("doc_type", "receipt")
    if doc_type == "invoice":
        return InvoiceGroundTruth(**data)
    return ReceiptGroundTruth(**data)


def save_ground_truth(gt: ReceiptGroundTruth | InvoiceGroundTruth, file_path: Path | str) -> Path:
    """Serialize and save a ground-truth object to disk in UTF-8 JSON format."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(gt.model_dump_json(indent=2))
    return path
