"""OCR package — Tesseract-based pipeline pipeline with EasyOCR fallback."""
from ocr.ocr_engine import configure_tesseract, ocr_text, ocr_file, ocr_json
from ocr.utils import extract_cbc, extract_cbc_consensus, resolve_value
from ocr.preprocessing import load_image

__all__ = [
    "configure_tesseract",
    "ocr_text", "ocr_file", "ocr_json",
    "extract_cbc", "extract_cbc_consensus", "resolve_value",
    "load_image",
]

