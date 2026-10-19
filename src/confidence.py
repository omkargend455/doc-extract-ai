"""Composite Review Decider and evidence-grounded confidence scoring for DocExtract AI."""

from __future__ import annotations

import re
from typing import Any, Optional
from pydantic import BaseModel

from src.schemas import DocumentValidationSummary, ReviewDecision, ReviewStatus
from src.validators import parse_date_to_iso, infer_doc_type, to_dict


def compute_field_confidence(
    field_name: str,
    value: Any,
    doc_type: str,
    validation_summary: Optional[DocumentValidationSummary] = None,
    raw_text: Optional[str] = None,
    currency: Optional[str] = None,
) -> tuple[float, str]:
    """Compute an evidence-based confidence heuristic and explanation for a single field.

    Confidence is treated strictly as a transparent heuristic based on verifiable signals,
    never as an authoritative guarantee of correctness.
    """
    if value is None or str(value).strip() == "":
        is_jurisdictional_tax = field_name in {"sgst_amount", "cgst_amount", "igst_amount"}
        if is_jurisdictional_tax:
            is_non_inr = currency is not None and str(currency).upper() != "INR"
            has_no_gst_mention = bool(
                raw_text and not re.search(r"\b(?:GST|SGST|CGST|IGST)\b", raw_text, re.IGNORECASE)
            )
            if is_non_inr or has_no_gst_mention:
                return 1.0, f"Jurisdictional tax field '{field_name}' (Indian GST) is not applicable to non-GST document."
            return 0.0, f"Jurisdictional tax component '{field_name}' not specified in document."

        if field_name == "discount_amount":
            return 0.85, f"Optional field '{field_name}' not specified; zero default assumed."

        return 0.0, f"Field '{field_name}' is missing or empty."

    val_str = str(value).strip()

    # 1. Date fields
    if "date" in field_name:
        iso_date = parse_date_to_iso(val_str)
        if not iso_date:
            return 0.35, f"Date '{val_str}' was extracted but could not be parsed into a calendar date."

        year = int(iso_date.split("-")[0])
        if not (2000 <= year <= 2035):
            return 0.40, f"Date '{iso_date}' has year {year} which falls outside plausible operational range."

        # Check chronology error if applicable
        if validation_summary:
            chronology_err = any(e.rule_name == "date_chronology" for e in validation_summary.errors)
            if chronology_err:
                return 0.50, f"Date '{iso_date}' is valid format but failed chronological consistency (invoice date > due date)."

        return 0.95, f"Date matches standard calendar format ({iso_date}) within expected year bounds."

    # 2. Total / Amount fields
    if any(k in field_name for k in ["total", "subtotal", "tax", "discount", "sgst", "cgst", "igst"]):
        try:
            cleaned = (
                val_str.replace("$", "")
                .replace("RM", "")
                .replace("₹", "")
                .replace("€", "")
                .replace("£", "")
                .replace(",", "")
                .strip()
            )
            amt = float(cleaned)
        except ValueError:
            return 0.20, f"Amount '{val_str}' contains non-numeric tokens."

        is_tax_field = any(k in field_name for k in ["tax", "sgst", "cgst", "igst"])
        is_discount_field = "discount" in field_name
        is_total_field = field_name in {"total", "total_amount"}

        # Total amount must be strictly positive (> 0.0)
        if is_total_field and amt <= 0.0:
            return 0.25, f"Total amount ({amt:.2f}) is non-positive, which violates accounting rules."

        # Subtotal, tax, and discount cannot be negative (< 0.0)
        if amt < 0.0:
            return 0.25, f"Monetary amount ({amt:.2f}) is negative, which violates accounting rules."

        # If arithmetic validation passed or failed, adjust confidence evidence
        if validation_summary:
            arithmetic_err = any(e.rule_name == "arithmetic_consistency" for e in validation_summary.errors)
            if arithmetic_err:
                if is_tax_field and amt == 0.0:
                    return 0.45, f"Zero tax ({amt:.2f}) parsed, but failed arithmetic cross-check (Subtotal + Tax != Total)."
                return 0.45, f"Amount ({amt:.2f}) parsed, but failed arithmetic cross-check (Subtotal + Tax != Total)."

            arithmetic_ok = any(
                r.rule_name == "arithmetic_consistency" and r.passed and "matches Stated Total" in r.message
                for r in validation_summary.rule_results
            )
            if arithmetic_ok:
                if is_tax_field and amt == 0.0:
                    return 0.98, f"Zero tax ({amt:.2f}) verified mathematically against subtotal and total balance."
                if is_discount_field and amt == 0.0:
                    return 0.98, f"Zero discount ({amt:.2f}) verified mathematically against subtotal and total balance."
                return 0.98, f"Amount ({amt:.2f}) verified mathematically against subtotal and tax line items."

        # Standalone handling when arithmetic check did not run or was skipped:
        if is_tax_field and amt == 0.0:
            zero_tax_explicit = bool(
                raw_text and re.search(
                    r"(?:0%|0\.00%|tax\s*[:(]?\s*0%|tax\s*[:(]?\s*\$?\s*0\.00|zero\s*rated|tax\s*exempt|\bexempt\b)",
                    raw_text,
                    re.IGNORECASE,
                )
            )
            if zero_tax_explicit:
                return 0.90, f"Explicit zero tax ({amt:.2f}) corroborated by zero-rate or exemption text in document."
            return 0.70, f"Zero tax ({amt:.2f}) extracted as numeric value, but lacks arithmetic cross-check or explicit zero-rate notation."

        if is_discount_field and amt == 0.0:
            return 0.85, f"Zero discount ({amt:.2f}) standard default for non-discounted invoice."

        return 0.88, f"Amount ({amt:.2f}) parsed successfully as positive decimal currency."

    # 3. Company / Vendor name
    if field_name in {"company", "vendor_name"}:
        if len(val_str) < 3:
            return 0.30, f"Vendor name '{val_str}' is suspiciously short (< 3 characters)."
        if re.search(r"\b(invoice|receipt|tax|total|cash|order)\b", val_str.lower()):
            return 0.50, f"Vendor name contains generic document keyword tokens ('{val_str}')."
        return 0.85, f"Vendor name '{val_str}' identified with clean formatting and valid length."

    # 4. Address field
    if field_name == "address":
        street_indicators = [
            r"\blot\b", r"\bjalan\b", r"\bjln\b", r"\bstreet\b", r"\broad\b",
            r"\btaman\b", r"\blorong\b", r"\bsuite\b", r"\bfloor\b", r"\bave\b",
            r"\bcity\b", r"\bstate\b", r"\bpostcode\b", r"\b\d{5}\b",
        ]
        has_keywords = any(re.search(pat, val_str.lower()) for pat in street_indicators)
        if has_keywords and len(val_str) >= 15:
            return 0.90, "Address contains recognizable street/geographic tokens and sufficient length."
        if len(val_str) >= 10:
            return 0.70, "Address has adequate length but lacks strong geographical keywords."
        return 0.40, f"Address '{val_str}' is unusually brief."

    # 5. Invoice number
    if field_name == "invoice_number":
        if re.match(r"^[A-Za-z0-9\-_/#]{3,30}$", val_str):
            return 0.90, f"Invoice number '{val_str}' conforms to standard alphanumeric identifier format."
        return 0.60, f"Invoice number '{val_str}' contains unexpected characters or irregular length."

    # 6. Currency field
    if field_name == "currency":
        known_currencies = {"USD", "EUR", "INR", "MYR", "GBP", "CAD", "AUD", "SGD", "JPY", "CHF"}
        if val_str.upper() in known_currencies:
            return 0.95, f"Standard currency ISO code ({val_str.upper()}) recognized."
        return 0.75, f"Currency identifier '{val_str}' extracted."

    # Fallback for generic text fields
    return 0.75, f"Field '{field_name}' populated with non-empty string."


def compute_composite_confidence(
    fields: dict[str, Any],
    doc_type: str,
    validation_summary: DocumentValidationSummary,
    raw_text: Optional[str] = None,
) -> tuple[float, dict[str, float], dict[str, str]]:
    """Calculate field-level confidences and overall document confidence."""
    field_confidences: dict[str, float] = {}
    evidence: dict[str, str] = {}

    # Define weights by document type
    if doc_type == "invoice":
        weights = {
            "total_amount": 0.30,
            "vendor_name": 0.20,
            "invoice_date": 0.20,
            "subtotal": 0.15,
            "due_date": 0.10,
            "invoice_number": 0.05,
        }
    else:
        weights = {
            "total": 0.35,
            "company": 0.25,
            "date": 0.25,
            "address": 0.15,
        }

    # Ensure all extracted fields have transparent confidence and evidence entries
    all_field_names = list(weights.keys())
    for k in fields.keys():
        if k not in all_field_names:
            all_field_names.append(k)

    doc_currency = fields.get("currency")
    for field_name in all_field_names:
        val = fields.get(field_name)
        score, explanation = compute_field_confidence(
            field_name=field_name,
            value=val,
            doc_type=doc_type,
            validation_summary=validation_summary,
            raw_text=raw_text,
            currency=doc_currency,
        )
        field_confidences[field_name] = round(score, 3)
        evidence[field_name] = explanation

    # Compute weighted sum based on core schema weights
    weighted_sum = sum(field_confidences[f] * w for f, w in weights.items() if f in field_confidences)
    total_weights = sum(w for f, w in weights.items() if f in field_confidences)
    raw_overall = weighted_sum / total_weights if total_weights else 0.0

    # Validation penalty: if errors exist, scale down overall confidence
    if validation_summary.errors:
        penalty = min(0.40, len(validation_summary.errors) * 0.15)
        raw_overall = max(0.10, raw_overall - penalty)

    overall_confidence = round(raw_overall, 3)
    return overall_confidence, field_confidences, evidence


def determine_review_routing(
    fields: dict[str, Any] | BaseModel,
    validation_summary: DocumentValidationSummary,
    doc_type: Optional[str] = None,
    raw_text: Optional[str] = None,
) -> ReviewDecision:
    """Determine human-in-the-loop review routing based on confidence and business rules.

    Routing States:
      - AUTO_APPROVED: High confidence (>= 0.85) and zero validation errors/warnings.
      - FLAGGED_WITH_WARNINGS: Acceptable confidence with minor warnings or missing non-critical fields.
      - HUMAN_REVIEW_REQUIRED: Validation errors (arithmetic mismatch, invalid date, negative total)
        or low overall confidence (< 0.70).
    """
    fields_dict = to_dict(fields)
    resolved_type = infer_doc_type(fields_dict, doc_type)

    overall_conf, field_confs, evidence = compute_composite_confidence(
        fields=fields_dict,
        doc_type=resolved_type,
        validation_summary=validation_summary,
        raw_text=raw_text,
    )

    reasons: list[str] = []

    # 1. Critical Hard Failures (Mandatory Human Review)
    if validation_summary.errors:
        for err in validation_summary.errors:
            reasons.append(f"Validation Error: {err.message}")
        return ReviewDecision(
            status=ReviewStatus.HUMAN_REVIEW_REQUIRED,
            overall_confidence=overall_conf,
            field_confidences=field_confs,
            evidence=evidence,
            reasons=reasons,
            requires_human_attention=True,
        )

    # 2. Low Confidence Failure
    if overall_conf < 0.70:
        reasons.append(f"Low overall extraction confidence ({overall_conf * 100:.1f}% < 70.0% threshold).")
        return ReviewDecision(
            status=ReviewStatus.HUMAN_REVIEW_REQUIRED,
            overall_confidence=overall_conf,
            field_confidences=field_confs,
            evidence=evidence,
            reasons=reasons,
            requires_human_attention=True,
        )

    # 3. Soft Warnings (Flagged for Optional Verification)
    if validation_summary.warnings or (0.70 <= overall_conf < 0.85):
        for warn in validation_summary.warnings:
            reasons.append(f"Validation Warning: {warn.message}")
        if overall_conf < 0.85:
            reasons.append(f"Moderate confidence ({overall_conf * 100:.1f}%), manual spot check advised.")
        return ReviewDecision(
            status=ReviewStatus.FLAGGED_WITH_WARNINGS,
            overall_confidence=overall_conf,
            field_confidences=field_confs,
            evidence=evidence,
            reasons=reasons,
            requires_human_attention=True,
        )

    # 4. Clean Automatic Approval
    reasons.append(f"All business rules passed ({validation_summary.passed_rules}/{validation_summary.total_rules}) with high confidence ({overall_conf * 100:.1f}%).")
    return ReviewDecision(
        status=ReviewStatus.AUTO_APPROVED,
        overall_confidence=overall_conf,
        field_confidences=field_confs,
        evidence=evidence,
        reasons=reasons,
        requires_human_attention=False,
    )
