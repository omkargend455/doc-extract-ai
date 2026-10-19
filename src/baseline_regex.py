"""Track A: Deterministic Regex and Heuristic Baseline Extractor for SROIE receipts."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional


class RegexBaselineExtractor:
    """Deterministic, rule-based baseline extractor for receipt documents.

    Extracts candidates and predicts the 4 core SROIE fields:
      - company: Store / vendor name
      - date: Transaction date
      - address: Store street address
      - total: Final transaction total
    """

    # Headers and noise to filter when looking for company name
    HEADER_NOISE = {
        "tax invoice", "invoice", "receipt", "official receipt",
        "simplified tax invoice", "cash receipt", "cash bill",
        "welcome", "thank you", "duplicate", "copy", "original",
        "order", "table", "sales receipt", "bill",
    }

    # Malaysian / common receipt address keywords
    ADDRESS_KEYWORDS = [
        r"\blot\b", r"\bjalan\b", r"\bjln\b",
        r"\bstreet\b", r"\broad\b", r"\btaman\b", r"\blorong\b",
        r"\bkampung\b", r"\bkawasan\b", r"\bperindustrian\b",
        r"\bbandar\b", r"\btingkat\b", r"\bblok\b", r"\bseksyen\b",
        r"\barea\b", r"\bnagar\b", r"\bdistrict\b", r"\bsuite\b",
        r"\bselangor\b", r"\bkuala lumpur\b", r"\bjohor\b", r"\bpenang\b",
        r"\bpulau pinang\b", r"\bperak\b", r"\bkedah\b", r"\bmelaka\b",
        r"\bnegeri sembilan\b", r"\bpahang\b", r"\bkelantan\b",
        r"\bsarawak\b", r"\bsabah\b", r"\bmalaysia\b",
        r"\b\d{5}\b",  # 5-digit Malaysian postal code
        r"(?<!receipt\s)(?<!invoice\s)(?<!inv\s)\bno\.?\s*\d+\b",
    ]

    # Date regex patterns
    DATE_PATTERNS = [
        # DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, YYYY-MM-DD, DD/MM/YY
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",
        r"\b(\d{1,2}\.\d{1,2}\.\d{2,4})\b",
        # DD Mon YYYY (e.g. 27 Apr 2018, 05-OCT-2016)
        r"\b(\d{1,2}[-\s]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-\s]+\d{2,4})\b",
        # Mon DD, YYYY
        r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b",
    ]

    # Money amount patterns (supports 1 to 9 digits, optional commas, and INR/RM/USD/EUR/GBP symbols)
    AMOUNT_PATTERN = re.compile(
        r"(?:RM|\$|USD|INR|₹)?\s*([0-9]{1,9}(?:,[0-9]{2,3})*\.[0-9]{2})\b",
        re.IGNORECASE
    )

    TOTAL_POSITIVE_KEYWORDS = [
        r"\bgrand\s*total\b",
        r"\btotal\s*amount\b",
        r"\btotal\s*incl\.?\s*gst\b",
        r"\btotal\s*\(incl\.?\s*gst\b",
        r"\btotal\s*rm\b",
        r"\bnet\s*total\b",
        r"\bjumlah\b",
        r"\btotal\b",
        r"\bamount\s*due\b",
        r"\bbalance\s*due\b",
    ]

    TOTAL_NEGATIVE_KEYWORDS = [
        "subtotal", "sub-total", "sub total",
        "total qty", "total item", "total items", "qty",
        "discount", "change", "rounding", "cash",
        "tender", "tendered", "tax", "gst", "service charge",
    ]

    TOTAL_EXCLUDES = [
        r"\btotal\s*gst\b", r"\bgst\s*total\b", r"\btotal\s*tax\b", r"\btax\s*total\b",
        r"\btotal\s*savings?\b", r"\btotal\s*discount\b", r"\bdiscount\s*total\b",
        r"\btotal\s*item\b", r"\btotal\s*qty\b", r"\btotal\s*quantity\b",
        r"\bexcluding\s*gst\b", r"\bexcluded\s*gst\b", r"\bexcl\.?\s*gst\b",
        r"\bsubtotal\b", r"\bsub-total\b", r"\bsub\s*total\b",
        r"\bchange\b", r"\bcash\b", r"\btender\b", r"\bbalance\b",
        r"\bincluded\s*in\s*total\b", r"\btax\s*included\b",
        r"\baccepted\s*total\b",
    ]

    PRIORITY_TOTAL_PATTERNS = [
        r"\btotal\s*rounded\b", r"\brounded\s*total\b", r"\bround\s*total\b",
        r"\bnet\s*total\b", r"\bnett\s*total\b", r"\bnet\s*amt\b", r"\bnet\s*amount\b",
        r"\btotal\s*sales\s*\(inclusive", r"\btotal\s*incl", r"\btotal\s*\(incl",
        r"\bgrand\s*total\b", r"\btotal\s*amount\b", r"\btotal\s*payable\b", r"\btotal\s*due\b",
        r"\btotal\s*rm\b", r"\bjumlah\b", r"\btotal\b",
    ]

    def extract_candidates(self, ocr_text: str) -> dict[str, Any]:
        """Extract all candidate text segments and matches from raw OCR text."""
        raw_lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]

        date_candidates: list[str] = []
        for line in raw_lines:
            for pattern in self.DATE_PATTERNS:
                for match in re.finditer(pattern, line, re.IGNORECASE):
                    date_candidates.append(match.group(1).strip())

        amount_candidates: list[str] = []
        for line in raw_lines:
            for match in self.AMOUNT_PATTERN.finditer(line):
                amount_candidates.append(match.group(1).replace(",", "").strip())

        total_lines: list[tuple[int, str]] = []
        for idx, line in enumerate(raw_lines):
            line_lower = line.lower()
            if any(re.search(kw, line_lower) for kw in self.TOTAL_POSITIVE_KEYWORDS):
                total_lines.append((idx, line))

        address_lines: list[tuple[int, str]] = []
        for idx, line in enumerate(raw_lines):
            line_lower = line.lower()
            if any(re.search(kw, line_lower) for kw in self.ADDRESS_KEYWORDS):
                address_lines.append((idx, line))

        return {
            "raw_lines": raw_lines,
            "date_candidates": date_candidates,
            "amount_candidates": amount_candidates,
            "total_lines": total_lines,
            "address_lines": address_lines,
        }

    def predict_company(self, raw_lines: list[str]) -> Optional[str]:
        """Predict company or vendor name from header lines."""
        # 1. Look for explicit 'Sold By:' or 'Vendor:' anchor lines
        for line in raw_lines[:15]:
            m = re.search(r"\b(?:sold\s*by|vendor|seller|merchant)\s*[:\-]?\s*(.+)", line, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                cleaned = re.sub(r"[,|\-_=*~#\s]+$", "", cand).strip()
                if len(cleaned) >= 3 and not any(kw in cleaned.lower() for kw in ["gstin", "invoice"]):
                    return cleaned

        # 2. Scan top header lines, skipping noise and disclaimers
        for line in raw_lines[:10]:
            line_clean = line.strip()
            line_lower = line_clean.lower()

            # Ignore pure separators, very short strings, or empty lines
            if len(line_clean) < 3 or re.match(r"^[-=_*#\s|]+$", line_clean):
                continue

            # Ignore table header lines, line items, and product rows
            if re.search(r"\b(?:description|gross\s*amount|taxable\s*val|sgst|cgst|igst|qty|total\s*₹|sac:|hsn:|imei|serial\s*no)\b", line_lower):
                continue
            if len(self.AMOUNT_PATTERN.findall(line_clean)) >= 2:
                continue

            # Ignore total lines and lines containing monetary amounts
            if any(re.search(kw, line_lower) for kw in self.TOTAL_POSITIVE_KEYWORDS):
                continue
            if self.AMOUNT_PATTERN.search(line_clean):
                continue

            # Ignore generic receipt headers and disclaimers (e.g., E. & O.E.)
            if line_lower in self.HEADER_NOISE:
                continue

            if re.search(r"\b(e\.\s*&\s*o\.\s*e|e&oe|page\s*\d+\s*of\s*\d+|billing\s*address|order\s*id|ship\s*to|bill\s*to)\b", line_lower):
                continue

            # Ignore phone, fax, tax, registration numbers
            if re.search(r"\b(tel|phone|fax|gst|reg|roc|ssm|co\.\s*no|tax\s*id|pan:)\b", line_lower):
                continue

            # Ignore lines that are pure dates or times
            if re.match(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", line_clean):
                continue

            # Clean leading/trailing artifacts
            cleaned = re.sub(r"^[|\-_=*~#\s]+|[|\-_=*~#\s]+$", "", line_clean).strip()
            if len(cleaned) >= 3:
                return cleaned

        # 3. Look for explicit corporate entity names anywhere in document (e.g., signature or seller blocks)
        for line in raw_lines:
            line_clean = line.strip()
            line_lower = line_clean.lower()
            if re.search(r"\b(?:description|gross\s*amount|taxable\s*val|sgst|cgst|igst|invoice|order|sac:|hsn:)\b", line_lower):
                continue
            m = re.search(
                r"\b([A-Z0-9][A-Za-z0-9\s.,&'\-]{2,60}?\b(?:Private\s+Limited|Pvt\.?\s*Ltd\.?|Limited|Ltd\.?|LLC|Inc\.?|Corporation|Corp\.?|Sdn\.?\s*Bhd\.?|Bhd\.?))\b",
                line_clean,
            )
            if m:
                cand = m.group(1).strip()
                cleaned = re.sub(r"^[|\-_=*~#\s]+|[|\-_=*~#\s]+$", "", cand).strip()
                if len(cleaned) >= 3 and cleaned.lower() not in self.HEADER_NOISE:
                    return cleaned

        return raw_lines[0] if raw_lines else None

    def _fix_ocr_year_typos(self, date_str: str) -> str:
        """Correct unambiguous OCR digit confusions in 4-digit years (e.g. 7018 -> 2018, 2617 -> 2017)."""
        fixed = re.sub(r"\b701([0-9])\b", r"201\1", date_str)
        fixed = re.sub(r"\b261([0-9])\b", r"201\1", fixed)
        return fixed

    def predict_date(self, raw_lines: list[str], date_candidates: list[str]) -> Optional[str]:
        """Predict the primary receipt date."""
        # 1. Look for date matches on lines with 'date', 'time', or 'tarikh'
        for line in raw_lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in ["date", "time", "tarikh", "dt", "d/t"]):
                for pattern in self.DATE_PATTERNS:
                    m = re.search(pattern, line, re.IGNORECASE)
                    if m:
                        candidate = self._fix_ocr_year_typos(m.group(1).strip())
                        if self._is_plausible_date(candidate):
                            return candidate

        # 2. Fall back to first plausible date candidate found in document
        for cand in date_candidates:
            fixed_cand = self._fix_ocr_year_typos(cand)
            if self._is_plausible_date(fixed_cand):
                return fixed_cand

        return date_candidates[0] if date_candidates else None

    def _is_plausible_date(self, text: str) -> bool:
        """Filter out obvious false-positive date matches."""
        digits = re.findall(r"\d+", text)
        if len(digits) == 3:
            try:
                nums = [int(d) for d in digits]
                # Year should be plausible (e.g. 2000-2035 or 00-35 for 2-digit)
                has_year = any((2000 <= n <= 2035) or (10 <= n <= 35) for n in nums)
                has_day = any(1 <= n <= 31 for n in nums)
                has_month = any(1 <= n <= 12 for n in nums)
                return has_year and has_day and has_month
            except ValueError:
                return False
        return len(digits) >= 2

    def predict_address(self, raw_lines: list[str], address_lines: list[tuple[int, str]]) -> Optional[str]:
        """Predict the store street address by combining header address lines."""
        if not address_lines:
            return None

        # Collect address candidate lines appearing in the top section (lines 1 to 10)
        neg_address_keywords = [
            "total", "subtotal", "gst", "tax invoice", "receipt no", "rcp-",
            "invoice no", "table", "cashier", "order", "date:", "tel:", "fax:"
        ]
        header_address_lines = [
            line for idx, line in address_lines
            if 0 < idx <= 10 and not any(kw in line.lower() for kw in neg_address_keywords)
        ]

        if header_address_lines:
            # Join multiple adjacent address lines, trimming trailing separators
            combined = ", ".join(line.rstrip(", \t") for line in header_address_lines[:4])
            # Clean OCR pipes or artifacts
            cleaned = re.sub(r"[|~_]+", " ", combined)
            # Collapse multiple commas
            cleaned = re.sub(r"[,;\s]+,", ", ", cleaned)
            return re.sub(r"\s+", " ", cleaned).strip(" ,;")

        # Fallback to the first matching address line
        return address_lines[0][1].strip()

    def predict_total(self, raw_lines: list[str], total_lines: list[tuple[int, str]], amount_candidates: list[str]) -> Optional[str]:
        """Predict the final total amount paid with hierarchical priority and adjacent line lookahead."""
        candidates: list[tuple[int, int, float]] = []

        for idx, line in enumerate(raw_lines):
            line_lower = line.lower()
            if any(re.search(ex, line_lower) for ex in self.TOTAL_EXCLUDES):
                continue
            for p_idx, pat in enumerate(self.PRIORITY_TOTAL_PATTERNS):
                if re.search(pat, line_lower):
                    amts = self.AMOUNT_PATTERN.findall(line)
                    if amts:
                        candidates.append((p_idx, idx, float(amts[-1].replace(",", ""))))
                    else:
                        # Multiline: look ahead 1-2 lines for adjacent amount
                        for offset in [1, 2]:
                            if idx + offset < len(raw_lines):
                                nxt = raw_lines[idx + offset]
                                nxt_lower = nxt.lower()
                                if any(re.search(ex, nxt_lower) for ex in [r"\bcash\b", r"\btender\b", r"\bchange\b", r"\bsubtotal\b", r"\btax\s*invoice\b"]):
                                    break
                                nxt_amts = self.AMOUNT_PATTERN.findall(nxt)
                                if nxt_amts:
                                    candidates.append((p_idx, idx, float(nxt_amts[-1].replace(",", ""))))
                                    break
                    break

        if candidates:
            # Sort by priority rank first (lowest p_idx), then later line position (-idx)
            candidates.sort(key=lambda c: (c[0], -c[1]))
            return f"{candidates[0][2]:.2f}"

        # Fallback to standard candidate amounts
        if amount_candidates:
            try:
                floats = [(float(a), a) for a in amount_candidates]
                floats.sort(key=lambda x: x[0], reverse=True)
                return f"{floats[0][0]:.2f}"
            except ValueError:
                return amount_candidates[-1]

        return None

    def extract_invoice(self, text: str) -> dict[str, Any]:
        """Deterministic baseline extractor for invoice documents."""
        candidates = self.extract_candidates(text)
        raw_lines = candidates["raw_lines"]

        vendor_name = self.predict_company(raw_lines)
        invoice_date = self.predict_date(raw_lines, candidates["date_candidates"])

        # Predict invoice number
        invoice_number = None
        for line in raw_lines:
            m = re.search(
                r"\b(?:invoice|inv)\s*(?:no\.?|num\.?|#|number)?\s*[:#\-]?\s*([A-Za-z0-9\-]+)\b|\bbill\s*(?:no\.?|num\.?|#|number)\s*[:#\-]?\s*([A-Za-z0-9\-]+)\b",
                line,
                re.IGNORECASE,
            )
            if m:
                cand = (m.group(1) or m.group(2) or "").strip()
                if len(cand) >= 3 and cand.lower() not in {"date", "to", "for", "address", "ing"}:
                    invoice_number = cand
                    break

        # Predict due date
        due_date = None
        for line in raw_lines:
            if re.search(r"\bdue\s*date\b", line, re.IGNORECASE):
                for pat in self.DATE_PATTERNS:
                    m = re.search(pat, line, re.IGNORECASE)
                    if m:
                        due_date = m.group(1).strip()
                        break
            if due_date:
                break

    def _extract_tax_breakdown(
        self, raw_lines: list[str]
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Extract taxable subtotal, taxes (SGST, CGST, IGST, total tax), and discounts from invoice lines.

        Subtotal is consistently defined as the pre-tax taxable value (base amount before taxes and discounts).
        Extracts SGST/UTGST and CGST separately internally where possible, then maps their sum to tax_amount.
        Does not mistake an individual line item's tax for the invoice's total tax.
        Recognizes tax-inclusive printed grand totals and avoids adding tax again.
        """
        subtotal: Optional[float] = None
        tax_amount: Optional[float] = None
        sgst_amount: Optional[float] = None
        cgst_amount: Optional[float] = None
        igst_amount: Optional[float] = None
        discount_amount: Optional[float] = None

        # 1. Check for tabular GST breakdown (e.g. Indian GST invoices)
        header_idx = None
        col_names = []
        col_defs = [
            ("gross", r"\bgross\s*amount\b"),
            ("discount", r"\b(?:discounts?|coupons?)(?:/(?:discounts?|coupons?))*\b"),
            ("taxable", r"\btaxable\s*val(?:ue)?\b"),
            ("sgst", r"\bsgst(?:/utgst)?\b"),
            ("cgst", r"\bcgst\b"),
            ("igst", r"\bigst\b"),
            ("total", r"\btotal\b"),
        ]

        for idx, line in enumerate(raw_lines):
            line_lower = line.lower()
            if re.search(r"\btaxable\s*val", line_lower) and re.search(r"\b(?:sgst|cgst|igst|tax)\b", line_lower):
                header_idx = idx
                matches = []
                for name, pat in col_defs:
                    for m in re.finditer(pat, line, re.IGNORECASE):
                        matches.append((m.start(), name))
                matches.sort(key=lambda x: x[0])
                col_names = [m[1] for m in matches]
                break

        if header_idx is not None and col_names:
            for line in raw_lines[header_idx + 1:]:
                # Check for table summary row starting with Total or Grand Total
                if re.match(r"^(?:total|grand\s*total)\b", line.strip(), re.IGNORECASE):
                    amounts = [float(x.replace(",", "")) for x in self.AMOUNT_PATTERN.findall(line)]
                    if len(amounts) == len(col_names):
                        mapped = dict(zip(col_names, amounts))
                        subtotal = mapped.get("taxable")
                        discount_amount = mapped.get("discount", 0.0)
                        sgst_amount = mapped.get("sgst")
                        cgst_amount = mapped.get("cgst")
                        igst_amount = mapped.get("igst")
                        if sgst_amount is not None and cgst_amount is not None:
                            tax_amount = round(sgst_amount + cgst_amount, 2)
                        elif igst_amount is not None:
                            tax_amount = round(igst_amount, 2)
                        return subtotal, tax_amount, sgst_amount, cgst_amount, igst_amount, discount_amount

        # 2. Check for arithmetic relationship on Total line (e.g. Total 2 448.00 0.00 379.66 34.17 34.17 448.00)
        for line in raw_lines:
            if re.match(r"^(?:total|grand\s*total)\b", line.strip(), re.IGNORECASE):
                amounts = [float(x.replace(",", "")) for x in self.AMOUNT_PATTERN.findall(line)]
                if len(amounts) >= 3:
                    # Look for taxable + sgst + cgst == total where sgst == cgst
                    for i, g in enumerate(amounts):
                        for j, s in enumerate(amounts):
                            if i == j or s >= g:
                                continue
                            for k, t1 in enumerate(amounts):
                                if k == i or k == j or t1 <= 0:
                                    continue
                                for m, t2 in enumerate(amounts):
                                    if m == i or m == j or m == k:
                                        continue
                                    if abs(t1 - t2) < 0.01 and abs((s + t1 + t2) - g) < 0.05:
                                        subtotal = s
                                        sgst_amount = t1
                                        cgst_amount = t2
                                        tax_amount = round(t1 + t2, 2)
                                        discount_amount = 0.0
                                        return subtotal, tax_amount, sgst_amount, cgst_amount, igst_amount, discount_amount

        # 3. Check for explicit summary lines with labels (avoiding line-item rows)
        for line in raw_lines:
            line_clean = line.strip()
            line_lower = line_clean.lower()
            # Ignore line items (containing SAC, HSN, item codes)
            if re.search(r"\b(?:sac:|hsn:|qty\b)", line_lower):
                continue

            # Subtotal / Taxable value
            if subtotal is None and re.search(r"\b(?:sub-?total|taxable\s*value|taxable\s*amount)\b", line_lower):
                amounts = self.AMOUNT_PATTERN.findall(line_clean)
                if amounts:
                    try:
                        subtotal = float(amounts[-1].replace(",", ""))
                    except ValueError:
                        pass

            # Explicit SGST
            if sgst_amount is None and re.search(r"\b(?:sgst(?:/utgst)?)\b", line_lower) and not re.search(r"%\s*$", line_clean):
                amounts = self.AMOUNT_PATTERN.findall(line_clean)
                if amounts:
                    try:
                        sgst_amount = float(amounts[-1].replace(",", ""))
                    except ValueError:
                        pass

            # Explicit CGST
            if cgst_amount is None and re.search(r"\b(?:cgst)\b", line_lower) and not re.search(r"%\s*$", line_clean):
                amounts = self.AMOUNT_PATTERN.findall(line_clean)
                if amounts:
                    try:
                        cgst_amount = float(amounts[-1].replace(",", ""))
                    except ValueError:
                        pass

            # Explicit IGST
            if igst_amount is None and re.search(r"\b(?:igst)\b", line_lower) and not re.search(r"%\s*$", line_clean):
                amounts = self.AMOUNT_PATTERN.findall(line_clean)
                if amounts:
                    try:
                        igst_amount = float(amounts[-1].replace(",", ""))
                    except ValueError:
                        pass

            # General Tax / VAT
            if tax_amount is None and re.search(r"\b(?:tax|vat|gst)\b", line_lower) and not re.search(r"\btax\s*invoice\b", line_lower):
                amounts = self.AMOUNT_PATTERN.findall(line_clean)
                if amounts:
                    try:
                        tax_amount = float(amounts[-1].replace(",", ""))
                    except ValueError:
                        pass

            # Discount
            if discount_amount is None and re.search(r"\bdiscount\b", line_lower):
                amounts = self.AMOUNT_PATTERN.findall(line_clean)
                if amounts:
                    try:
                        discount_amount = float(amounts[-1].replace(",", ""))
                    except ValueError:
                        pass

        # Map sum of SGST and CGST if both present
        if sgst_amount is not None and cgst_amount is not None:
            tax_amount = round(sgst_amount + cgst_amount, 2)
        elif igst_amount is not None and tax_amount is None:
            tax_amount = round(igst_amount, 2)

        return subtotal, tax_amount, sgst_amount, cgst_amount, igst_amount, discount_amount

    def extract_invoice(self, text: str) -> dict[str, Any]:
        """Deterministic baseline extractor for invoice documents."""
        candidates = self.extract_candidates(text)
        raw_lines = candidates["raw_lines"]

        vendor_name = self.predict_company(raw_lines)
        invoice_date = self.predict_date(raw_lines, candidates["date_candidates"])

        # Predict invoice number
        invoice_number = None
        for line in raw_lines:
            m = re.search(
                r"\b(?:invoice|inv)\s*(?:no\.?|num\.?|#|number)?\s*[:#\-]?\s*([A-Za-z0-9\-]+)\b|\bbill\s*(?:no\.?|num\.?|#|number)\s*[:#\-]?\s*([A-Za-z0-9\-]+)\b",
                line,
                re.IGNORECASE,
            )
            if m:
                cand = (m.group(1) or m.group(2) or "").strip()
                if len(cand) >= 3 and cand.lower() not in {"date", "to", "for", "address", "ing"}:
                    invoice_number = cand
                    break

        # Predict due date
        due_date = None
        for line in raw_lines:
            if re.search(r"\bdue\s*date\b", line, re.IGNORECASE):
                for pat in self.DATE_PATTERNS:
                    m = re.search(pat, line, re.IGNORECASE)
                    if m:
                        due_date = m.group(1).strip()
                        break
            if due_date:
                break

        # Extract taxable subtotal, taxes (SGST, CGST, total tax), and discount
        subtotal, tax_amount, sgst_amount, cgst_amount, igst_amount, discount_amount = self._extract_tax_breakdown(raw_lines)

        # Predict total
        total_str = self.predict_total(raw_lines, candidates["total_lines"], candidates["amount_candidates"])
        total_amount = None
        if total_str:
            try:
                total_amount = float(total_str)
            except ValueError:
                total_amount = None

        # Predict currency with word boundaries (prevents 'mumbai' matching 'rm')
        currency = "USD"
        if re.search(r"(?:₹|\b(?:inr|rs\.?)\b)", text, re.IGNORECASE):
            currency = "INR"
        elif re.search(r"\b(?:rm|myr)\b", text, re.IGNORECASE):
            currency = "MYR"
        elif re.search(r"(?:€|\b(?:eur|euro)\b)", text, re.IGNORECASE):
            currency = "EUR"
        elif re.search(r"(?:£|\b(?:gbp)\b)", text, re.IGNORECASE):
            currency = "GBP"

        return {
            "vendor_name": vendor_name,
            "invoice_number": invoice_number,
            "invoice_date": invoice_date,
            "due_date": due_date,
            "subtotal": subtotal,
            "tax_amount": tax_amount,
            "sgst_amount": sgst_amount,
            "cgst_amount": cgst_amount,
            "igst_amount": igst_amount,
            "discount_amount": discount_amount if discount_amount is not None else 0.0,
            "total_amount": total_amount,
            "currency": currency,
        }

    def _preprocess_coordinates(self, text: str) -> str:
        """Reconstruct 2D reading order from OCR coordinate bounding boxes if present."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return text
        boxes = []
        is_coord = False
        for line in lines:
            parts = line.split(",", 8)
            if len(parts) >= 9:
                try:
                    boxes.append((int(parts[1]), int(parts[0]), parts[8]))
                    is_coord = True
                except ValueError:
                    boxes.append((0, 0, line))
            else:
                boxes.append((0, 0, line))
        if not is_coord:
            return text
        boxes.sort(key=lambda b: (b[0], b[1]))
        reconstructed = []
        curr_line = []
        curr_y = None
        for y, x, t in boxes:
            if curr_y is None or abs(y - curr_y) <= 16:
                curr_line.append((x, t))
                if curr_y is None:
                    curr_y = y
            else:
                curr_line.sort(key=lambda b: b[0])
                reconstructed.append(" ".join(w for _, w in curr_line))
                curr_line = [(x, t)]
                curr_y = y
        if curr_line:
            curr_line.sort(key=lambda b: b[0])
            reconstructed.append(" ".join(w for _, w in curr_line))
        return "\n".join(reconstructed)

    def extract(self, ocr_text: str, doc_type: str = "receipt") -> dict[str, Any]:
        """Run candidate extraction and predict target fields for receipts or invoices."""
        clean_ocr_text = self._preprocess_coordinates(ocr_text)

        if doc_type == "invoice":
            return self.extract_invoice(clean_ocr_text)

        candidates = self.extract_candidates(clean_ocr_text)
        raw_lines = candidates["raw_lines"]

        company = self.predict_company(raw_lines)
        date = self.predict_date(raw_lines, candidates["date_candidates"])
        address = self.predict_address(raw_lines, candidates["address_lines"])
        total = self.predict_total(raw_lines, candidates["total_lines"], candidates["amount_candidates"])

        return {
            "company": company,
            "date": date,
            "address": address,
            "total": total,
        }
