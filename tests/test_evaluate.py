"""Unit tests for evaluation metrics (exact match, normalized match, macro metrics)."""

from src.evaluate import (
    normalize_company,
    normalize_date,
    normalize_address,
    normalize_total,
    match_field,
    evaluate_predictions,
)


def test_normalizers():
    """Verify normalization helpers for company, date, address, and total."""
    # Company
    assert normalize_company("MR. D.I.Y. (KUCHAI) SDN BHD") == "mr d i y kuchai"
    assert normalize_company("BOOK KEEPER SDN. BHD.") == "book keeper"

    # Date
    assert normalize_date("27/04/2018") == "2018-04-27"
    assert normalize_date("2018-04-27") == "2018-04-27"
    assert normalize_date("27-04-18") == "2018-04-27"
    assert normalize_date("07-06-16") == "2016-06-07"

    # Address
    assert "balakong" in normalize_address("KAWASAN PERINDUSTRIAN BALAKONG, 43300")

    # Total
    assert normalize_total("RM 13.30") == 13.30
    assert normalize_total("$ 40.00") == 40.00
    assert normalize_total("1,250.50") == 1250.50
    assert normalize_total(None) is None


def test_match_field_exact_and_normalized():
    """Verify match_field distinction between exact and normalized matches."""
    # Total with currency symbol difference
    exact, norm_m, score = match_field("13.30", "RM 13.30", "total")
    assert not exact
    assert norm_m

    # Date format difference: 27/04/2018 vs 2018-04-27
    exact, norm_m, score = match_field("27/04/2018", "2018-04-27", "date")
    assert not exact
    assert norm_m

    # Company with SDN BHD suffix difference
    exact, norm_m, score = match_field("Book Keeper", "BOOK KEEPER SDN BHD", "company")
    assert not exact
    assert norm_m
    assert score > 70.0


def test_evaluate_predictions_aggregation():
    """Verify dataset evaluation and macro summary computation."""
    preds = [
        {"company": "BOOK KEEPER", "date": "27/04/2018", "address": "Jalan Damansara", "total": "40.00"},
        {"company": "Apex Supplies", "date": "01/01/2026", "address": "Boston MA", "total": "220.00"},
    ]
    gts = [
        {"company": "BOOK KEEPER", "date": "2018-04-27", "address": "NO 42 JALAN DAMANSARA", "total": "40.00"},
        {"company": "Apex Supplies", "date": "2026-01-01", "address": "Boston MA", "total": "220.00"},
    ]

    res = evaluate_predictions(preds, gts)

    assert res["total_documents"] == 2
    assert "summary_df" in res
    assert "details_df" in res

    summary = res["summary_df"]
    # Both totals match exactly
    total_row = summary[summary["Field"] == "total"].iloc[0]
    assert total_row["Exact Match (%)"] == 100.0
    assert total_row["Normalized Match (%)"] == 100.0

    # Macro averages present
    macro_row = summary[summary["Field"] == "MACRO AVERAGE"].iloc[0]
    assert macro_row["Normalized Match (%)"] > 50.0
