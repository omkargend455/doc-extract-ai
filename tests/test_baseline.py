"""Unit tests for Track A deterministic regex baseline extractor."""

from src.baseline_regex import RegexBaselineExtractor


def test_baseline_extraction_on_standard_receipt():
    """Verify extraction of company, date, address, and total on structured receipt text."""
    sample_text = """
    CYNBIT COFFEE ROASTERS SDN BHD
    14 INDUSTRIAL AREA, MALVIYA NAGAR
    JAIPUR, RAJASTHAN 302017
    TEL: +91 141 555 0199
    === separator ===
    RECEIPT NO: RCP-88321
    DATE: 18/10/2026 09:42
    CASHIER: AMIT
    === separator ===
    1 ESPRESSO SINGLE         4.50
    1 OAT MILK LATTE          6.00
    TAX                       1.00
    === separator ===
    TOTAL AMOUNT: $11.50
    CASH TENDERED:   $20.00
    CHANGE:           $8.50
    """
    extractor = RegexBaselineExtractor()
    preds = extractor.extract(sample_text)

    assert preds["company"] is not None
    assert "CYNBIT COFFEE ROASTERS" in preds["company"]
    assert preds["date"] == "18/10/2026"
    assert preds["address"] is not None
    assert "MALVIYA NAGAR" in preds["address"]
    assert preds["total"] == "11.50"


def test_baseline_extraction_on_sroie_ocr_snippet():
    """Verify extraction on real OCR output snippet from sample X51006619862."""
    ocr_snippet = """
    MR. D.I.Y. (KUCHAI) SDN BHD
    LOT 1851-A & 1851-B, JALAN KPB 6,
    KAWASAN PERINDUSTRIAN BALAKONG,
    43300 SERI KEMBANGAN, SELANGOR.
    (GST ID No: 0000473792512)

    HANDPHONE STICKER WNY-037
    DECO STICKER  11.20
    2.10

    Total Incl. GST@6% = RM 13.30
    CASH : RM 13.30
    GST @6% included in total = RM 0.75

    07-06-16 15:26
    OPERATOR: SITI AMINAH
    """
    extractor = RegexBaselineExtractor()
    preds = extractor.extract(ocr_snippet)

    assert "MR. D.I.Y." in (preds["company"] or "")
    assert preds["date"] == "07-06-16"
    assert "BALAKONG" in (preds["address"] or "")
    assert preds["total"] == "13.30"


def test_baseline_empty_and_noise_handling():
    """Verify extractor returns None gracefully when given empty or pure noise text."""
    extractor = RegexBaselineExtractor()
    empty_preds = extractor.extract("")
    assert empty_preds["company"] is None
    assert empty_preds["date"] is None
    assert empty_preds["address"] is None
    assert empty_preds["total"] is None

    noise_text = "------\n=====\n***\nTAX INVOICE\nRECEIPT\nTEL: 123456\n"
    noise_preds = extractor.extract(noise_text)
    assert noise_preds["date"] is None
    assert noise_preds["total"] is None


def test_candidates_inspection():
    """Verify extract_candidates returns populated lists for debugging."""
    text = "TAX INVOICE\nSTORE ABC\nDATE: 01/01/2026\nJALAN AMPANG, KUALA LUMPUR\nTOTAL: $50.00"
    extractor = RegexBaselineExtractor()
    candidates = extractor.extract_candidates(text)

    assert len(candidates["raw_lines"]) == 5
    assert "01/01/2026" in candidates["date_candidates"]
    assert "50.00" in candidates["amount_candidates"]
    assert len(candidates["total_lines"]) == 1
    assert len(candidates["address_lines"]) >= 1
