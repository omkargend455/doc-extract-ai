"""Deterministic business rules and cross-field validators for DocExtract AI."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel

from src.schemas import DocumentValidationSummary, ValidationRuleResult


DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d/%m/%y",
    "%d-%m-%Y",
    "%d-%m-%y",
    "%d.%m.%Y",
    "%d.%m.%y",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%d-%b-%Y",
    "%b %d, %Y",
    "%B %d, %Y",
]


def parse_date_to_iso(date_str: Any) -> Optional[str]:
    """Parse a date string into ISO-8601 (YYYY-MM-DD) format."""
    if not date_str:
        return None
    s = str(date_str).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        try:
            datetime.strptime(s, "%Y-%m-%d")
            return s
        except ValueError:
            return None

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def to_dict(data: dict[str, Any] | BaseModel) -> dict[str, Any]:
    """Convert BaseModel or dict into a plain Python dictionary."""
    if isinstance(data, BaseModel):
        return data.model_dump()
    return dict(data)


def infer_doc_type(data: dict[str, Any], explicit_type: Optional[str] = None) -> str:
    """Infer document type ('receipt' or 'invoice') from data contents."""
    if explicit_type in {"receipt", "invoice"}:
        return explicit_type
    doc_type = data.get("doc_type")
    if doc_type in {"receipt", "invoice"}:
        return str(doc_type)
    if "vendor_name" in data or "invoice_number" in data or "subtotal" in data:
        return "invoice"
    return "receipt"


def validate_required_fields(data: dict[str, Any], doc_type: str = "receipt") -> list[ValidationRuleResult]:
    """Verify presence of all mandatory fields for the document type."""
    results: list[ValidationRuleResult] = []

    if doc_type == "invoice":
        mandatory = [
            ("vendor_name", "Vendor name is required for invoice documents."),
            ("invoice_date", "Invoice date is required for accounting records."),
            ("total_amount", "Final total amount is required for invoice processing."),
        ]
    else:
        mandatory = [
            ("company", "Company / store name is required for receipt documents."),
            ("date", "Transaction date is required for expense tracking."),
            ("total", "Receipt total amount is required for reimbursement."),
        ]

    for field_name, err_msg in mandatory:
        val = data.get(field_name)
        passed = (val is not None) and (str(val).strip() != "")
        results.append(
            ValidationRuleResult(
                rule_name=f"required_field_{field_name}",
                passed=passed,
                severity="ERROR" if not passed else "INFO",
                message=err_msg if not passed else f"Required field '{field_name}' is present.",
                field=field_name,
                details={"value": val},
            )
        )

    return results


def validate_positive_amounts(data: dict[str, Any], doc_type: str = "receipt") -> list[ValidationRuleResult]:
    """Verify that currency totals and monetary amounts are positive numbers (> 0.0)."""
    results: list[ValidationRuleResult] = []

    # 1. Total amount check
    total_key = "total_amount" if doc_type == "invoice" else "total"
    total_val = data.get(total_key)

    if total_val is not None:
        try:
            total_float = float(total_val)
            passed = total_float > 0.0
            results.append(
                ValidationRuleResult(
                    rule_name="positive_total_amount",
                    passed=passed,
                    severity="ERROR" if not passed else "INFO",
                    message="Total amount must be strictly positive (> 0.0)." if not passed else f"Total amount ({total_float:.2f}) is positive.",
                    field=total_key,
                    details={"total": total_float},
                )
            )
        except (ValueError, TypeError):
            results.append(
                ValidationRuleResult(
                    rule_name="positive_total_amount",
                    passed=False,
                    severity="ERROR",
                    message=f"Total amount '{total_val}' is not a valid numeric value.",
                    field=total_key,
                    details={"raw_value": str(total_val)},
                )
            )

    # 2. Subtotal check if present
    subtotal_val = data.get("subtotal")
    if subtotal_val is not None:
        try:
            subtotal_float = float(subtotal_val)
            passed = subtotal_float >= 0.0
            results.append(
                ValidationRuleResult(
                    rule_name="non_negative_subtotal",
                    passed=passed,
                    severity="ERROR" if not passed else "INFO",
                    message="Subtotal cannot be negative." if not passed else f"Subtotal ({subtotal_float:.2f}) is non-negative.",
                    field="subtotal",
                    details={"subtotal": subtotal_float},
                )
            )
        except (ValueError, TypeError):
            results.append(
                ValidationRuleResult(
                    rule_name="non_negative_subtotal",
                    passed=False,
                    severity="ERROR",
                    message=f"Subtotal '{subtotal_val}' is not a valid numeric value.",
                    field="subtotal",
                )
            )

    return results


def validate_date_plausibility(data: dict[str, Any], doc_type: str = "receipt") -> list[ValidationRuleResult]:
    """Verify that extracted dates can be parsed and fall within a realistic year range (2000-2035)."""
    results: list[ValidationRuleResult] = []

    date_keys = ["invoice_date", "due_date"] if doc_type == "invoice" else ["date"]

    for key in date_keys:
        val = data.get(key)
        if val is None or str(val).strip() == "":
            continue

        iso_date = parse_date_to_iso(val)
        if not iso_date:
            results.append(
                ValidationRuleResult(
                    rule_name=f"date_format_{key}",
                    passed=False,
                    severity="ERROR",
                    message=f"Date '{val}' for field '{key}' could not be parsed into a standard calendar date.",
                    field=key,
                    details={"raw_value": str(val)},
                )
            )
            continue

        year = int(iso_date.split("-")[0])
        passed_range = 2000 <= year <= 2035
        results.append(
            ValidationRuleResult(
                rule_name=f"date_plausibility_{key}",
                passed=passed_range,
                severity="ERROR" if not passed_range else "INFO",
                message=f"Date '{iso_date}' has year {year} which is outside the plausible range (2000-2035)." if not passed_range else f"Date '{iso_date}' is plausible.",
                field=key,
                details={"parsed_iso": iso_date, "year": year},
            )
        )

    return results


def validate_date_chronology(data: dict[str, Any]) -> list[ValidationRuleResult]:
    """Verify that invoice_date <= due_date. Evaluated ONLY when both dates are present."""
    inv_date_raw = data.get("invoice_date")
    due_date_raw = data.get("due_date")

    # If either date is missing, do not guess or perform chronological check
    if not inv_date_raw or not due_date_raw:
        return [
            ValidationRuleResult(
                rule_name="date_chronology",
                passed=True,
                severity="INFO",
                message="Chronological validation skipped: both invoice_date and due_date must be present.",
                field="due_date",
            )
        ]

    iso_inv = parse_date_to_iso(inv_date_raw)
    iso_due = parse_date_to_iso(due_date_raw)

    if not iso_inv or not iso_due:
        return [
            ValidationRuleResult(
                rule_name="date_chronology",
                passed=False,
                severity="WARNING",
                message="Could not evaluate chronology: one or both dates failed formatting validation.",
                field="due_date",
                details={"invoice_date": inv_date_raw, "due_date": due_date_raw},
            )
        ]

    passed = iso_inv <= iso_due
    return [
        ValidationRuleResult(
            rule_name="date_chronology",
            passed=passed,
            severity="ERROR" if not passed else "INFO",
            message=f"Chronology mismatch: invoice date ({iso_inv}) is later than due date ({iso_due})." if not passed else f"Chronology valid: invoice date ({iso_inv}) <= due date ({iso_due}).",
            field="due_date",
            details={"invoice_date": iso_inv, "due_date": iso_due},
        )
    ]


def validate_arithmetic_consistency(data: dict[str, Any]) -> list[ValidationRuleResult]:
    """Verify arithmetic cross-consistency: Subtotal + Tax - Discount == Total.

    Subtotal Definition:
      Subtotal is consistently defined as the pre-tax taxable value (the net taxable base
      amount upon which taxes are assessed, before taxes and after line discounts).
      For tax-inclusive invoices (such as Indian GST invoices), subtotal is the Taxable Value,
      NOT the tax-inclusive gross amount or final total. The printed grand total is tax-inclusive.

    Strict rule: Only performed when necessary numeric components are present.
    Does NOT infer missing subtotal, tax, or discount values.
    """
    subtotal_val = data.get("subtotal")
    total_val = data.get("total_amount") if data.get("total_amount") is not None else data.get("total")

    # Condition: Requires at least subtotal and total to evaluate arithmetic relation
    if subtotal_val is None or total_val is None:
        return [
            ValidationRuleResult(
                rule_name="arithmetic_consistency",
                passed=True,
                severity="INFO",
                message="Arithmetic check skipped: subtotal or total amount not present.",
                field="total_amount",
            )
        ]

    try:
        subtotal = float(subtotal_val)
        total = float(total_val)
    except (ValueError, TypeError):
        return [
            ValidationRuleResult(
                rule_name="arithmetic_consistency",
                passed=False,
                severity="ERROR",
                message="Arithmetic check failed: subtotal or total is non-numeric.",
                field="total_amount",
            )
        ]

    # Components present without inference: support tax_amount or explicit SGST/CGST/IGST breakdown
    tax_raw = data.get("tax_amount")
    if tax_raw is None:
        sgst_comp = data.get("sgst_amount")
        cgst_comp = data.get("cgst_amount")
        igst_comp = data.get("igst_amount")
        if sgst_comp is not None or cgst_comp is not None or igst_comp is not None:
            tax_raw = round(
                (float(sgst_comp or 0.0) + float(cgst_comp or 0.0) + float(igst_comp or 0.0)),
                2,
            )

    discount_raw = data.get("discount_amount")

    has_tax = tax_raw is not None
    has_discount = discount_raw is not None

    tax = float(tax_raw) if has_tax else 0.0
    discount = float(discount_raw) if has_discount else 0.0

    expected_total = round(subtotal + tax - discount, 2)
    delta = round(abs(expected_total - total), 2)
    tolerance = 0.05

    passed = delta <= tolerance

    details = {
        "subtotal": subtotal,
        "tax_amount": float(tax_raw) if has_tax else None,
        "discount_amount": float(discount_raw) if has_discount else None,
        "expected_total": expected_total,
        "stated_total": total,
        "delta": delta,
    }

    if passed:
        message = f"Arithmetic consistency passed: Subtotal ({subtotal:.2f})"
        if has_tax:
            message += f" + Tax ({tax:.2f})"
        if has_discount:
            message += f" - Discount ({discount:.2f})"
        message += f" = Expected {expected_total:.2f} matches Stated Total {total:.2f}."
        return [
            ValidationRuleResult(
                rule_name="arithmetic_consistency",
                passed=True,
                severity="INFO",
                message=message,
                field="total_amount",
                details=details,
            )
        ]

    # If tax was not provided, and subtotal != total, flag for human review rather than inventing values
    if not has_tax:
        return [
            ValidationRuleResult(
                rule_name="arithmetic_consistency",
                passed=False,
                severity="ERROR",
                message=f"Subtotal ({subtotal:.2f}) does not equal Total ({total:.2f}), and tax breakdown is missing. Document requires human review.",
                field="total_amount",
                details=details,
            )
        ]

    # Hard arithmetic failure when tax/discount were declared but numbers do not sum up
    return [
        ValidationRuleResult(
            rule_name="arithmetic_consistency",
            passed=False,
            severity="ERROR",
            message=f"Arithmetic mismatch: Subtotal ({subtotal:.2f}) + Tax ({tax:.2f}) - Discount ({discount:.2f}) = {expected_total:.2f}, but Stated Total is {total:.2f} (delta: {delta:.2f}).",
            field="total_amount",
            details=details,
        )
    ]


def validate_document(data: dict[str, Any] | BaseModel, doc_type: Optional[str] = None) -> DocumentValidationSummary:
    """Run all relevant business-rule validators on a document and return a comprehensive summary."""
    doc_dict = to_dict(data)
    resolved_type = infer_doc_type(doc_dict, doc_type)

    all_results: list[ValidationRuleResult] = []

    # 1. Required fields
    all_results.extend(validate_required_fields(doc_dict, resolved_type))

    # 2. Positive total & monetary amounts
    all_results.extend(validate_positive_amounts(doc_dict, resolved_type))

    # 3. Date calendar plausibility
    all_results.extend(validate_date_plausibility(doc_dict, resolved_type))

    # 4. Date chronology (invoices only)
    if resolved_type == "invoice":
        all_results.extend(validate_date_chronology(doc_dict))

    # 5. Arithmetic consistency
    all_results.extend(validate_arithmetic_consistency(doc_dict))

    errors = [r for r in all_results if not r.passed and r.severity == "ERROR"]
    warnings = [r for r in all_results if not r.passed and r.severity == "WARNING"]
    passed_rules = [r for r in all_results if r.passed]

    return DocumentValidationSummary(
        is_valid=(len(errors) == 0),
        total_rules=len(all_results),
        passed_rules=len(passed_rules),
        failed_rules=len(errors) + len(warnings),
        errors=errors,
        warnings=warnings,
        rule_results=all_results,
    )
