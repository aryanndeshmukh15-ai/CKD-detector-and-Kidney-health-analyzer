"""
ocr/postprocessing.py
---------------------
Post-processing utilities for OCR output.
Handles merging broken bounding boxes, cleaning common text errors, and normalization.
"""

import re
import logging
from typing import List, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

COMMON_MISTAKES = {
    "O": "0", "o": "0",
    "l": "1", "I": "1",
    "B": "8",
    "Z": "2", "z": "2",
}

def clean_text(text: str, is_numeric: bool = False) -> str:
    text = text.strip()
    if not text:
        return ""
    if is_numeric:
        corrected = []
        for char in text:
            if char in COMMON_MISTAKES:
                corrected.append(COMMON_MISTAKES[char])
            else:
                corrected.append(char)
        text = "".join(corrected)
        text = re.sub(r'[^\d\.,\-]', '', text)
    else:
        text = re.sub(r'\s{2,}', ' ', text)
    return text


def merge_boxes(words: List[Dict[str, Any]], line_tolerance: int = 15) -> List[Dict[str, Any]]:
    """
    Merge bounding boxes into visual rows using spatial Y-coordinate grouping.

    WHY we ignore Tesseract block/line grouping:
    Lab reports are multi-column. Tesseract assigns labels (left col) and values
    (right col) to DIFFERENT blocks. Native grouping produces separate lines for
    label and value, so extract_cbc never finds the value next to its label.

    Spatial Y-band grouping puts ALL words in the same vertical band on one line:
        y~210: "Hemoglobin (Hb)  12.5  Low  13.0-17.0  g/dL"  ← correct
    """
    if not words:
        return []

    heights = [w.get('height', 15) for w in words if w.get('height', 0) > 0]
    bucket = max(10, int((sum(heights) / len(heights)) * 0.6)) if heights else 15

    grouped = defaultdict(list)
    for w in words:
        bucket_key = (w['top'] // bucket) * bucket
        grouped[bucket_key].append(w)

    merged_output = []
    for key in sorted(grouped.keys()):
        line = sorted(grouped[key], key=lambda w: w['left'])
        if not line:
            continue
        left   = min(w['left'] for w in line)
        top    = min(w['top']  for w in line)
        right  = max(w['left'] + w['width']  for w in line)
        bottom = max(w['top']  + w['height'] for w in line)
        avg_conf = sum(w['conf'] for w in line) / len(line)
        text = " ".join(w['text'] for w in line)
        merged_output.append({
            "text":   clean_text(text),
            "conf":   avg_conf,
            "left":   left,
            "top":    top,
            "width":  right - left,
            "height": bottom - top,
        })

    return merged_output


def extract_text_from_merged(merged_lines: List[Dict[str, Any]]) -> str:
    return "\n".join([line['text'] for line in merged_lines])