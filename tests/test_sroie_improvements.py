"""Regression tests for SROIE baseline extraction improvements.

Verifies:
1. 2D spatial coordinate layout reconstruction.
2. Multiline and lookahead total extraction.
3. Total rounding priority over pre-rounding amounts.
4. Total negative keyword filtering (Total GST, Total Savings, Cash).
5. OCR digit confusion correction in dates (e.g., 7018 -> 2018).
6. Address comma normalization.
"""

from src.baseline_regex import RegexBaselineExtractor
from src.evaluate import match_field


def test_2d_spatial_coordinate_reconstruction():
    """Verify that 2D bounding boxes are sorted into natural reading order."""
    # Bounding boxes simulating OCR where column 1 is on left, column 2 on right
    coord_text = """
184,100,698,100,698,140,184,140,STORE NAME SDN BHD
70,300,420,300,420,340,70,340,TOTAL INCL. GST
673,301,841,301,841,341,673,341,RM 13.30
64,400,161,400,161,440,64,440,CASH
672,401,830,401,830,441,672,441,RM 50.00
"""
    extractor = RegexBaselineExtractor()
    preds = extractor.extract(coord_text, doc_type="receipt")

    assert preds["company"] == "STORE NAME SDN BHD"
    assert preds["total"] == "13.30"


def test_total_rounding_priority_over_tax_and_subtotal():
    """Verify TOTAL ROUNDED takes priority over pre-rounding or tax totals."""
    sample_text = """
    MY MART SDN BHD
    DATE: 11-06-2018
    TOTAL INCL. GST RM 11.23
    ROUNDING ADJUSTMENT RM 0.02
    TOTAL ROUNDED RM 11.25
    CASH RM 50.00
    CHANGE RM 38.75
    """
    extractor = RegexBaselineExtractor()
    preds = extractor.extract(sample_text, doc_type="receipt")
    assert preds["total"] == "11.25"


def test_total_negative_keyword_filtering():
    """Verify that Total GST, Total Savings, Total Items are never selected as receipt Total."""
    sample_text = """
    SUPERSTORE BHD
    DATE: 20/04/2018
    TOTAL QTY: 5
    TOTAL SALES (EXCLUDING GST) : 82.50
    TOTAL GST : 4.95
    TOTAL SALES (INCLUSIVE OF GST) : 87.45
    TOTAL SAVINGS : 10.00
    CASH : 100.00
    """
    extractor = RegexBaselineExtractor()
    preds = extractor.extract(sample_text, doc_type="receipt")
    assert preds["total"] == "87.45"


def test_multiline_total_lookahead():
    """Verify that when label and amount are on adjacent lines, amount is properly extracted."""
    multiline_text = """
    COFFEE SHOP
    DATE: 01/01/2018
    TOTAL AMOUNT:
    RM 24.50
    CASH
    RM 50.00
    """
    extractor = RegexBaselineExtractor()
    preds = extractor.extract(multiline_text, doc_type="receipt")
    assert preds["total"] == "24.50"


def test_ocr_year_typo_correction():
    """Verify that OCR confusion in 4-digit years (e.g., 7018 -> 2018) is corrected."""
    extractor = RegexBaselineExtractor()
    fixed_date = extractor._fix_ocr_year_typos("25/12/7018")
    assert fixed_date == "25/12/2018"

    preds = extractor.extract("STORE\nDATE: 30/07/2617\nTOTAL: 10.00")
    assert preds["date"] == "30/07/2017"


def test_address_comma_cleanup():
    """Verify that address multi-line concatenation does not produce duplicate commas."""
    sample_text = """
    BOOK STORE
    LOT 1851-A, JALAN KPB 6,
    KAWASAN PERINDUSTRIAN BALAKONG,
    43300 SERI KEMBANGAN, SELANGOR
    DATE: 01/01/2018
    TOTAL: 10.00
    """
    extractor = RegexBaselineExtractor()
    preds = extractor.extract(sample_text, doc_type="receipt")
    assert ",," not in preds["address"]
    assert "BALAKONG" in preds["address"]
