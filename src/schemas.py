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
    """Standardized ground-truth annotations for invoice documents.

    Subtotal Definition:
      Subtotal is consistently defined as the pre-tax taxable value (the net taxable base
      amount upon which taxes are assessed, before taxes and after line discounts).
      For tax-inclusive invoices (such as Indian GST invoices), subtotal is the Taxable Value,
      NOT the tax-inclusive gross amount or final total.
    """

    doc_id: str = Field(description="Unique document identifier")
    doc_type: str = Field(default="invoice", description="Document type identifier")
    vendor_name: Optional[str] = Field(default=None, description="Issuing company or vendor name")
    invoice_number: Optional[str] = Field(default=None, description="Unique invoice or bill identifier")
    invoice_date: Optional[str] = Field(default=None, description="Issue date in YYYY-MM-DD format")
    due_date: Optional[str] = Field(default=None, description="Payment due date in YYYY-MM-DD format")
    subtotal: Optional[float] = Field(
        default=None,
        description="Subtotal representing pre-tax taxable value (base amount before taxes and discounts)",
    )
    tax_amount: Optional[float] = Field(
        default=None,
        description="Total tax amount (mapped from sum of SGST+CGST or IGST, or general tax)",
    )
    sgst_amount: Optional[float] = Field(default=None, description="SGST / UTGST tax component")
    cgst_amount: Optional[float] = Field(default=None, description="CGST tax component")
    igst_amount: Optional[float] = Field(default=None, description="IGST tax component")
    discount_amount: Optional[float] = Field(default=0.0, description="Applied discount amount")
    total_amount: Optional[float] = Field(default=None, description="Final grand total payable (tax-inclusive printed total)")
    currency: Optional[str] = Field(default="USD", description="Currency code (USD, EUR, INR, MYR)")
    items: list[dict[str, Any]] = Field(default_factory=list, description="Line item details")
    expected_valid: bool = Field(default=True, description="Whether this document passes all business arithmetic/chronology rules")
    injected_errors: list[str] = Field(default_factory=list, description="Deliberate error flags for edge-case testing")
    raw_text: Optional[str] = Field(default=None, description="Raw text of the invoice")

    @field_validator(
        "subtotal", "tax_amount", "sgst_amount", "cgst_amount", "igst_amount", "discount_amount", "total_amount",
        mode="before"
    )
    @classmethod
    def normalize_invoice_amounts(cls, v: Any) -> Optional[float]:
        """Normalize numeric or string amounts into float."""
        if v is None or v == "":
            return None
        if isinstance(v, (int, float)):
            return float(v)
        cleaned = str(v).replace("$", "").replace("RM", "").replace("₹", "").replace("€", "").replace("£", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None

    def model_post_init(self, __context: Any) -> None:
        """Map SGST + CGST + IGST to tax_amount if tax_amount is not explicitly set."""
        if self.tax_amount is None:
            comp_sum = 0.0
            has_comp = False
            for c in (self.sgst_amount, self.cgst_amount, self.igst_amount):
                if c is not None:
                    comp_sum += float(c)
                    has_comp = True
            if has_comp:
                self.tax_amount = round(comp_sum, 2)


class ValidationRuleResult(BaseModel):
    """Result of a single deterministic validation rule check."""

    rule_name: str = Field(description="Unique rule identifier (e.g., arithmetic_consistency, date_chronology)")
    passed: bool = Field(description="Whether the rule passed successfully")
    severity: str = Field(default="ERROR", description="Rule severity: ERROR, WARNING, or INFO")
    message: str = Field(description="Human-readable explanation of the check outcome")
    field: Optional[str] = Field(default=None, description="Primary field associated with this rule")
    details: Optional[dict[str, Any]] = Field(default=None, description="Detailed computational diagnostics")


class DocumentValidationSummary(BaseModel):
    """Aggregated validation summary for a document across all business rules."""

    is_valid: bool = Field(description="True if no ERROR-level validation rules failed")
    total_rules: int = Field(default=0, description="Total number of evaluated rules")
    passed_rules: int = Field(default=0, description="Count of rules that passed")
    failed_rules: int = Field(default=0, description="Count of rules that failed")
    errors: list[ValidationRuleResult] = Field(default_factory=list, description="Rules that failed with ERROR severity")
    warnings: list[ValidationRuleResult] = Field(default_factory=list, description="Rules that failed with WARNING severity")
    rule_results: list[ValidationRuleResult] = Field(default_factory=list, description="Complete list of all rule outcomes")


class ReviewStatus(str):
    """Human-in-the-loop review routing states."""

    AUTO_APPROVED = "AUTO_APPROVED"
    FLAGGED_WITH_WARNINGS = "FLAGGED_WITH_WARNINGS"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"


class ReviewDecision(BaseModel):
    """Composite review decision combining confidence heuristics and validation outcomes."""

    status: str = Field(description="Routing outcome: AUTO_APPROVED, FLAGGED_WITH_WARNINGS, or HUMAN_REVIEW_REQUIRED")
    overall_confidence: float = Field(description="Evidence-based confidence score between 0.0 and 1.0")
    field_confidences: dict[str, float] = Field(default_factory=dict, description="Field-level heuristic confidence scores")
    evidence: dict[str, str] = Field(default_factory=dict, description="Transparent explanations for each field score")
    reasons: list[str] = Field(default_factory=list, description="Explanations for the review routing decision")
    requires_human_attention: bool = Field(description="Flag indicating manual human review is necessary")


class ExtractionResult(BaseModel):
    """Unified container for document extraction output from any engine."""

    doc_id: str = Field(description="Document identifier")
    doc_type: str = Field(default="receipt", description="Document type: receipt or invoice")
    source: str = Field(description="Extraction provenance: nvidia, baseline, or baseline_fallback")
    provider: Optional[str] = Field(default=None, description="Underlying provider: nvidia or regex")
    model_used: Optional[str] = Field(default=None, description="Model identifier used if LLM was successfully invoked")
    fields: dict[str, Any] = Field(default_factory=dict, description="Extracted key-value fields")
    raw_text: Optional[str] = Field(default=None, description="Underlying OCR or digital text")
    error_message: Optional[str] = Field(default=None, description="Sanitized diagnostics if fallback was triggered")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional provenance and input mode metadata")


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
