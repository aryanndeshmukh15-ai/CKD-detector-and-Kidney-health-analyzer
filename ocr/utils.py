"""
ocr/utils.py
------------
Utility functions for CBC value extraction from OCR text.
Handles lab report parsing, sanity validation, and value resolution.
"""

import re
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


# ── Physiological sanity ranges ───────────────────────────────────────────────

SANE_RANGES: Dict[str, Tuple[float, float]] = {
    "hemoglobin":  (3.0,    25.0),
    "rbc":         (1.0,    10.0),
    "pcv":         (10.0,   70.0),
    "mcv":         (50.0,   130.0),
    "mch":         (10.0,   50.0),
    "mchc":        (20.0,   40.0),
    "rdw":         (5.0,    30.0),
    "wbc":         (500.0,  100000.0),
    "platelets":   (10000,  1000000),
    "creatinine":  (0.3,    15.0),   # serum creatinine mg/dL — SG values like 1.011 excluded by range fingerprint
    "urea":        (1.0,    300.0),
    "sodium":      (100.0,  170.0),
    "potassium":   (1.0,    10.0),
    "glucose":     (30.0,   600.0),
}


def is_sane(name: str, value: float) -> bool:
    """
    Check if a value is within physiologically plausible range.

    Args:
        name: Marker key (e.g. 'hemoglobin').
        value: Numeric value to check.

    Returns:
        True if within range, False otherwise.
    """
    lo, hi = SANE_RANGES.get(name, (0, 1e9))
    return lo <= value <= hi


# ── CBC label patterns ────────────────────────────────────────────────────────

CBC_PATTERNS: Dict[str, list] = {
    "hemoglobin": [r"h[ae]moglobin", r"\bhb\b", r"\bhgb\b"],
    "rbc":        [r"\brbc\b", r"r\.b\.c", r"red\s+blood\s+cell", r"erythrocyte", r"rbc\s+count",
                   r"rbc.*count", r"red\s+cell\s+count"],
    "wbc":        [r"\bwbc\b", r"\bwec\b", r"w\.b\.c", r"white\s+blood\s+cell",
                   r"leucocyte", r"leukocyte", r"\btlc\b",
                   r"total\s+leukocyte", r"total\s+leucocyte",
                   r"w[be]c\s+count", r"wbc.*cells.*mm", r"wec.*cells.*mm"],
    "pcv":        [r"\bpcv\b", r"packed\s+cell", r"haematocrit", r"hematocrit", r"\bhct\b"],
    "mcv":        [r"\bmcv\b", r"mean\s+corp.*volume"],
    "mch":        [r"\bmch\b(?!c)", r"mean\s+corp.*hemo(?!globin\s+conc)"],
    "mchc":       [r"\bmchc\b", r"mean\s+corp.*hemo.*conc"],
    "rdw":        [r"\brdw\b", r"red\s+cell\s+dist", r"red\s+cell\s+distribution\s+width"],
    "platelets":  [r"platelet", r"\bplt\b"],
    "creatinine": [r"\bserum\s+creatinine\b", r"\bcreatinine\b", r"\bcreatinine,\s*mg",
                   r"\bcreat\b(?!\s+ratio)", r"\bscr\b"],
    "urea":       [r"\burea\b(?!\s+nitrogen)", r"\bbun\b", r"blood\s+urea", r"urea\s+nitrogen"],
    "sodium":     [r"\bsodium\b", r"\bna\+?\b(?!\w)", r"sodium.*mmol", r"na.*mmol",
                   r"Na\+"],
    "potassium":  [r"\bpotassium\b", r"\bk\+?\b(?!\s*\d{4})", r"potassium.*mmol",
                   r"K\+"],
    "glucose":    [r"\bglucose\b", r"\bblood\s+sugar\b", r"\bbgr\b", r"\bbs\b(?!\w)"],
}

# Matches reference ranges like "12.0 - 17.5" or "4,000–11,000"
_RANGE_RE = re.compile(r"[\d,]+\.?\d*\s*[-–—]\s*[\d,]+\.?\d*")


def extract_value(text: str, name: str) -> Optional[float]:
    """
    Extract the first physiologically sane number from a text string.
    Strips reference ranges before searching.
    """
    # Normalise comma-separated thousands: 7,800 → 7800
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)

    # Handle scientific notation: 7.8 x 10^3 → 7800
    text = re.sub(
        r"(\d+\.?\d*)\s*[xX×]\s*10\s*[\^(]?\s*3\s*[)]?",
        lambda m: str(float(m.group(1)) * 1000), text
    )
    text = re.sub(
        r"(\d+\.?\d*)\s*[xX×]\s*10\s*[\^(]?\s*6\s*[)]?",
        lambda m: str(float(m.group(1)) * 1e6), text
    )

    # Convert thou/mm3 (thousands) — strip ref ranges first, then find number
    # adjacent to 'thou' in either direction (before or after)
    if name in ("wbc", "platelets"):
        clean_for_thou = _RANGE_RE.sub(" ", text)
        if re.search(r"thou", clean_for_thou, re.IGNORECASE):
            # Find the last standalone decimal near 'thou' (before or after)
            nums = list(re.finditer(r"\b(\d+\.?\d*)\b", clean_for_thou))
            for m in reversed(nums):
                v = float(m.group(1))
                # Small floats in thou/mm3 range (1-100) mean value in thousands
                if 1.0 <= v <= 999.0:
                    converted = round(v * 1000)
                    if 500 <= converted <= 100000:
                        return converted


    # Strip reference ranges
    clean = _RANGE_RE.sub(" ", text)

    # Use word-boundary anchors: this prevents matching numbers embedded inside
    # unit strings like '1.73m2' (from 'mL/min/1.73m2' GFR units).
    for match in re.finditer(r"(?<![\w./])\d+\.?\d*(?![\w/])", clean):
        try:
            v = float(match.group())
            if is_sane(name, v):
                return v
        except ValueError:
            continue

    return None


def extract_cbc(text: str) -> Dict[str, float]:
    """
    Parse OCR text and extract all CBC / chemistry values.

    Two-pass strategy:
    1. Label matching — find lines with known CBC label patterns.
    2. Normal-range fingerprinting — identify markers by their reference range
       when labels are garbled (common in dense tabular reports).
    """
    lines = [re.sub(r"\s{2,}", " ", l.strip()) for l in text.split("\n") if l.strip()]
    extracted: Dict[str, float] = {}

    # ── Pass 1: label matching (forward + backward value search) ─────────────
    for i, line in enumerate(lines):
        ll = line.lower()

        # Skip lines that are clearly not direct marker readings
        if re.search(r'\bratio\b|\bindex\b', ll):
            continue
        for name, patterns in CBC_PATTERNS.items():
            if name in extracted:
                continue
            if not any(re.search(p, ll) for p in patterns):
                continue

            val = extract_value(line, name)

            # Search FORWARD individual lines
            if val is None:
                for j in range(i + 1, min(i + 7, len(lines))):
                    next_ll = lines[j].lower()
                    # Stop if we hit another CBC label
                    is_label = any(
                        any(re.search(p, next_ll) for p in pats)
                        for pats in CBC_PATTERNS.values()
                    )
                    if is_label and j > i + 1:
                        break
                    # Stop if we hit a clearly different test section (GFR, ratio lines)
                    if re.search(r'\bgfr\b|\bestimated\b|\bcategory\b|\bratio\b|\bindex\b', next_ll):
                        break
                    val = extract_value(lines[j], name)
                    if val is not None:
                        break

            # Search FORWARD merged window — handles value + unit on separate lines
            # e.g. "7.05\nthou/mm3" -> "7.05 thou/mm3" -> 7050
            if val is None:
                fwd_win = " ".join(lines[i + 1: min(i + 6, len(lines))])
                val = extract_value(fwd_win, name)
                if val is not None:
                    logger.debug(f"Pass1 forward-window: {name}={val}")

            # Search BACKWARD — handles multi-column PDFs where value appears
            # before the label (e.g. Dr Lal PathLabs column layout).
            # REVERSED so we check the line closest to the label first.
            if val is None:
                for j in reversed(range(max(0, i - 5), i)):
                    prev_ll = lines[j].lower()
                    is_label = any(
                        any(re.search(p, prev_ll) for p in pats)
                        for pats in CBC_PATTERNS.values()
                    )
                    if is_label:
                        continue
                    # Skip prose/header lines (2+ alphabetic words with a stray digit)
                    # e.g. "SwasthFit Super 4" — the 4 is not a lab value
                    alpha_words = sum(1 for w in lines[j].split() if re.search(r'[a-zA-Z]', w))
                    if alpha_words >= 2:
                        continue
                    val = extract_value(lines[j], name)
                    if val is not None:
                        logger.debug(f"Pass1 backward-single: {name}={val} line={j}")
                        break

            if val is None:
                # Merge backward window — handles units on separate lines
                win = " ".join(lines[max(0, i - 5):i])
                val = extract_value(win, name)
                if val is not None:
                    logger.debug(f"Pass1 backward-window: {name}={val}")


            if val is not None:
                extracted[name] = val
                logger.debug(f"Pass1: {name} = {val}")
            break

    # ── Pass 2: normal-range fingerprinting ───────────────────────────────────
    # Maps a regex matching the reference range to the marker name + value range
    RANGE_FINGERPRINTS = [
        ("hemoglobin", r"12[\.\s]*0?\s*[-–]\s*1[67][\.\s]*\d"),
        ("hemoglobin", r"11[\.\s]*5?\s*[-–]\s*1[67][\.\s]*\d"),
        ("wbc",        r"[34][\.,]?0{3}\s*[-–]\s*1[01][\.,]?0{3}"),
        ("wbc",        r"3[\.,]9\s*[-–]\s*11[\.,]7"),
        ("rbc",        r"3[\.\s]*[58]\s*[-–]\s*[56][\.\s]*\d"),
        ("rbc",        r"4[\.\s]*[02]\s*[-–]\s*[56][\.\s]*\d"),
        ("pcv",        r"3[56]\s*[-–]\s*5[05]"),
        ("sodium",     r"13[56]\s*[-–]\s*14[56]"),
        ("potassium",  r"3[\.\s]*[45]\s*[-–]\s*5[\.\s]*[01]"),
        # creatinine range is 0.6-1.3 — must NOT match specific gravity (1.001-1.030)
        ("creatinine", r"0[\.\s]*[67]\s*[-–]\s*1[\.\s]*[23]\b"),
        ("creatinine", r"0[\.\s]*7\s*[-–]\s*1[\.\s]*3\b"),
        ("urea",       r"[78]\s*[-–]\s*4[05]"),
        ("glucose",    r"7[05]\s*[-–]\s*1[01][05]"),
    ]

    for i, line in enumerate(lines):
        ll = line.lower()
        for name, range_pat in RANGE_FINGERPRINTS:
            if name in extracted:
                continue
            if not re.search(range_pat, ll):
                continue
            # Found a reference range — extract value from the SAME line only
            # (avoids picking up values from unrelated nearby lines)
            val = extract_value(line, name)
            if val is not None:
                extracted[name] = val
                logger.debug(f"Pass2 (range fingerprint): {name} = {val}")

    return extracted


def extract_cbc_consensus(texts: list) -> Dict[str, float]:
    """
    Run extract_cbc on multiple OCR text outputs (one per preprocessing
    candidate) and return a consensus dict where each marker's value is
    the one most frequently extracted across candidates.

    This neutralises single-candidate digit misreads (e.g. one pipeline
    reads RBC as 9.2 while three others correctly read 5.2).
    """
    from collections import Counter

    if not texts:
        return {}

    # Collect per-marker values across all texts
    all_extractions: Dict[str, list] = {}
    for text in texts:
        cbc = extract_cbc(text)
        for name, val in cbc.items():
            all_extractions.setdefault(name, []).append(val)

    # Majority vote per marker
    consensus: Dict[str, float] = {}
    for name, values in all_extractions.items():
        counter = Counter(values)
        winner, count = counter.most_common(1)[0]
        consensus[name] = winner
        if len(counter) > 1:
            logger.info(
                f"CBC consensus for '{name}': {dict(counter)} → winner {winner} "
                f"({count}/{len(values)} votes)"
            )

    return consensus


def resolve_value(
    cbc: Dict[str, float],
    cbc_key: str,
    manual_val: Optional[float],
    default: float,
    default_label: str,
) -> Tuple[float, str]:
    """
    Resolve a marker value from report, manual input, or default.

    Args:
        cbc: Extracted CBC dict from OCR.
        cbc_key: Key to look up in cbc dict.
        manual_val: Value entered manually by user (or None).
        default: Fallback default value.
        default_label: Human-readable label for the default (e.g. 'assumed 13.5').

    Returns:
        Tuple of (value, source) where source is 'report', 'manual', or default_label.
    """
    if cbc.get(cbc_key) is not None:
        return cbc[cbc_key], "report"
    if manual_val is not None:
        return manual_val, "manual"
    return default, default_label
