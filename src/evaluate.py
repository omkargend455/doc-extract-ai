"""Evaluation engine: field-level exact match, normalized match, and fuzzy metrics."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional
import pandas as pd
from rapidfuzz import fuzz


def normalize_company(text: Optional[str]) -> str:
    """Normalize company name: lowercase, strip punctuation, remove legal types."""
    if not text:
        return ""
    s = str(text).lower().strip()
    # Strip OCR noise and symbols
    s = re.sub(r"[^\w\s]", " ", s)
    # Remove common Malaysian and international company suffixes
    s = re.sub(r"\b(sdn\s*bhd|sdn|bhd|pte\s*ltd|ltd|inc|llc|co|corp)\b", "", s)
    return " ".join(s.split())


def normalize_date(text: Optional[str]) -> str:
    """Normalize date strings to ISO YYYY-MM-DD or standard compact format."""
    if not text:
        return ""
    s = str(text).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s

    formats = [
        "%d/%m/%Y", "%d/%m/%y",
        "%d-%m-%Y", "%d-%m-%y",
        "%d.%m.%Y", "%d.%m.%y",
        "%Y/%m/%d", "%Y-%m-%d",
        "%d %b %Y", "%d %B %Y",
        "%d-%b-%Y", "%d-%b-%y",
        "%d-%B-%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Fallback: keep alphanumeric characters only
    return re.sub(r"[^\w]", "", s.lower())


def normalize_address(text: Optional[str]) -> str:
    """Normalize address: lowercase, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    s = str(text).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def normalize_total(val: Any) -> Optional[float]:
    """Parse numeric total amount to standard float with 2 decimal places."""
    if val is None or val == "":
        return None
    if isinstance(val, (int, float)):
        return round(float(val), 2)
    s = str(val).replace("$", "").replace("RM", "").replace("rm", "").replace("₹", "").replace(",", "").strip()
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def match_field(pred_val: Any, gt_val: Any, field_name: str) -> tuple[bool, bool, float]:
    """Evaluate a single field prediction against ground truth.

    Returns:
        (exact_match, normalized_match, rapidfuzz_ratio)
    """
    pred_str = str(pred_val).strip() if pred_val is not None else ""
    gt_str = str(gt_val).strip() if gt_val is not None else ""

    # Exact Match
    exact = (pred_str == gt_str) and (len(gt_str) > 0)

    # Fuzzy similarity (0 to 100) using lowercased strings
    score = float(fuzz.token_set_ratio(pred_str.lower(), gt_str.lower())) if pred_str and gt_str else 0.0

    # Normalized Match
    norm_match = False
    if field_name == "total":
        p_num = normalize_total(pred_val)
        g_num = normalize_total(gt_val)
        norm_match = (p_num is not None) and (g_num is not None) and (abs(p_num - g_num) < 0.01)
    elif field_name == "date":
        p_date = normalize_date(pred_val)
        g_date = normalize_date(gt_val)
        norm_match = (p_date == g_date) and (len(g_date) > 0)
    elif field_name == "company":
        p_comp = normalize_company(pred_val)
        g_comp = normalize_company(gt_val)
        norm_match = (p_comp == g_comp and len(g_comp) > 0) or (score >= 85.0)
    elif field_name == "address":
        p_addr = normalize_address(pred_val)
        g_addr = normalize_address(gt_val)
        norm_match = (p_addr == g_addr and len(g_addr) > 0) or (score >= 80.0)
    else:
        norm_match = exact

    return exact, norm_match, score


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    fields: list[str] = ["company", "date", "address", "total"],
) -> dict[str, Any]:
    """Calculate exact match, normalized match, and RapidFuzz metrics across a dataset."""
    assert len(predictions) == len(ground_truths), "Predictions and ground truths count must match"
    total_docs = len(predictions)
    if total_docs == 0:
        return {"total_documents": 0, "summary_df": pd.DataFrame()}

    field_stats: dict[str, dict[str, float]] = {f: {"exact_count": 0, "norm_count": 0, "total_score": 0.0} for f in fields}

    detailed_records: list[dict[str, Any]] = []

    for idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        record: dict[str, Any] = {"doc_idx": idx}
        for f in fields:
            p_val = pred.get(f)
            g_val = gt.get(f)
            exact, norm_m, score = match_field(p_val, g_val, f)

            if exact:
                field_stats[f]["exact_count"] += 1
            if norm_m:
                field_stats[f]["norm_count"] += 1
            field_stats[f]["total_score"] += score

            record[f"{f}_pred"] = p_val
            record[f"{f}_gt"] = g_val
            record[f"{f}_exact"] = exact
            record[f"{f}_norm"] = norm_m
            record[f"{f}_fuzz"] = score

        detailed_records.append(record)

    summary_rows = []
    for f in fields:
        em_rate = (field_stats[f]["exact_count"] / total_docs) * 100.0
        nm_rate = (field_stats[f]["norm_count"] / total_docs) * 100.0
        avg_fuzz = field_stats[f]["total_score"] / total_docs
        summary_rows.append({
            "Field": f,
            "Exact Match (%)": round(em_rate, 2),
            "Normalized Match (%)": round(nm_rate, 2),
            "Avg RapidFuzz Score": round(avg_fuzz, 2),
        })

    # Macro Averages
    macro_em = sum(r["Exact Match (%)"] for r in summary_rows) / len(summary_rows)
    macro_nm = sum(r["Normalized Match (%)"] for r in summary_rows) / len(summary_rows)
    macro_fuzz = sum(r["Avg RapidFuzz Score"] for r in summary_rows) / len(summary_rows)

    summary_rows.append({
        "Field": "MACRO AVERAGE",
        "Exact Match (%)": round(macro_em, 2),
        "Normalized Match (%)": round(macro_nm, 2),
        "Avg RapidFuzz Score": round(macro_fuzz, 2),
    })

    summary_df = pd.DataFrame(summary_rows)
    details_df = pd.DataFrame(detailed_records)

    return {
        "total_documents": total_docs,
        "summary_df": summary_df,
        "details_df": details_df,
        "macro_exact_match": round(macro_em, 2),
        "macro_normalized_match": round(macro_nm, 2),
    }
