"""
ocr/pdf_extractor.py
--------------------
Robust PDF text extraction using PyMuPDF (fitz).

Strategy per page:
  1. Native text layer  – fast & perfect for digital PDFs  
  2. OCR via Tesseract  – for scanned / image-only pages
  3. Raw word list      – last resort PyMuPDF word grab

Also handles:
  - Encrypted PDFs (password support)
  - Damaged / partially-corrupted PDFs (lenient open)
  - Rotation-mismarked pages
"""

import io
import logging
from typing import Optional

import fitz           # PyMuPDF
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_MIN_NATIVE_CHARS = 20   # chars below this = treat page as image-only


def _page_to_numpy(page: fitz.Page, dpi: int = 300) -> np.ndarray:
    """Render a PDF page to a BGR numpy array (OpenCV / Tesseract ready)."""
    zoom = dpi / 72
    mat  = fitz.Matrix(zoom, zoom)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return np.array(img)[:, :, ::-1].copy()   # RGB → BGR


def extract_pdf_text(
    filepath: str,
    ocr_engine=None,
    dpi: int = 300,
    password: str = "",
) -> str:
    """
    Extract all text from a PDF, with automatic OCR fallback per page.

    Args:
        filepath:   Full path to the PDF file.
        ocr_engine: An OCREngine instance (created internally if None).
        dpi:        Resolution for rasterising pages when OCR is needed.
        password:   Password for encrypted PDFs.

    Returns:
        Combined extracted text from all pages.
    """
    if ocr_engine is None:
        from ocr.ocr_engine import OCREngine
        ocr_engine = OCREngine()

    all_text: list = []

    # ── Open with lenient / repair mode ─────────────────────────────────────
    try:
        doc = fitz.open(filepath)
    except Exception as first_err:
        logger.warning(f"[PDF] Normal open failed ({first_err}), trying repair mode…")
        try:
            # fitz repair: open raw bytes and let PyMuPDF fix the structure
            with open(filepath, "rb") as f:
                raw = f.read()
            doc = fitz.open(stream=raw, filetype="pdf")
        except Exception as e:
            logger.error(f"[PDF] Repair open also failed: {e}")
            return ""

    # ── Handle encryption ────────────────────────────────────────────────────
    if doc.is_encrypted:
        ok = doc.authenticate(password)
        if not ok:
            logger.warning("[PDF] Encrypted — trying empty password…")
            ok = doc.authenticate("")          # many lab PDFs have empty password
        if not ok:
            logger.error("[PDF] Cannot decrypt — skipping native layer, will OCR raw bytes via page rendering.")

    total_pages = doc.page_count
    logger.info(f"[PDF] Opened '{filepath}': {total_pages} page(s), encrypted={doc.is_encrypted}")

    for page_num in range(total_pages):
        page_text = ""

        try:
            page = doc[page_num]
        except Exception as e:
            logger.error(f"[PDF] Cannot access page {page_num+1}: {e}")
            continue

        # ── Strategy 1: Native text ──────────────────────────────────────────
        try:
            native = page.get_text("text").strip()
            if len(native) >= _MIN_NATIVE_CHARS:
                logger.debug(f"[PDF] Page {page_num+1}: native text ({len(native)} ch)")
                page_text = native
        except Exception as e:
            logger.warning(f"[PDF] Native text failed on page {page_num+1}: {e}")

        # ── Strategy 2: OCR ─────────────────────────────────────────────────
        if not page_text:
            try:
                logger.info(f"[PDF] Page {page_num+1}: rasterising for OCR…")
                img_arr = _page_to_numpy(page, dpi=dpi)
                page_text = ocr_engine.ocr_text(img_arr)
                logger.info(f"[PDF] Page {page_num+1}: OCR → {len(page_text)} chars")
            except Exception as e:
                logger.warning(f"[PDF] OCR failed on page {page_num+1}: {e}")

        # ── Strategy 3: Raw word list ────────────────────────────────────────
        if not page_text:
            try:
                words = page.get_text("words")
                words_sorted = sorted(words, key=lambda w: (round(w[1] / 10) * 10, w[0]))
                page_text = " ".join(w[4] for w in words_sorted).strip()
                if page_text:
                    logger.warning(f"[PDF] Page {page_num+1}: used raw word-list fallback.")
            except Exception as e:
                logger.error(f"[PDF] Word-list fallback failed: {e}")

        if page_text:
            all_text.append(page_text)

    doc.close()

    combined = "\n\n".join(all_text)
    logger.info(f"[PDF] extraction complete: {len(combined)} total chars")
    return combined
