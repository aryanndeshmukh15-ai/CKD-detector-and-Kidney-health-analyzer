"""
ocr/preprocessing.py
--------------------
Image preprocessing pipeline for OCR.
Handles grayscale, denoising, thresholding, deskewing, CLAHE, and resizing.
"""

import logging
import os
from typing import Optional, Dict
import cv2
import numpy as np
import pytesseract
import re

logger = logging.getLogger(__name__)

DEBUG_MODE = os.environ.get("OCR_DEBUG", "") == "1"
DEBUG_DIR = "ocr_debug"

if DEBUG_MODE:
    os.makedirs(DEBUG_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_debug(img: np.ndarray, step_name: str):
    """Save intermediate images if debug mode is active."""
    if DEBUG_MODE:
        path = os.path.join(DEBUG_DIR, f"{step_name}.jpg")
        cv2.imwrite(path, img)
        logger.debug(f"Saved debug image: {path}")

def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR or RGBA image to grayscale."""
    if len(img.shape) == 2:
        res = img
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _save_debug(res, "01_grayscale")
    return res

def _resize(img: np.ndarray, min_width: int = 2400) -> np.ndarray:
    """Upscale image so width >= min_width for better OCR accuracy."""
    h, w = img.shape[:2]
    if w < min_width:
        scale = min_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        _save_debug(img, "02_resized")
    return img

def _denoise(img: np.ndarray, strength: str = "medium") -> np.ndarray:
    """Remove noise using Gaussian or median blur."""
    if strength == "light":
        res = cv2.GaussianBlur(img, (3, 3), 0)
    elif strength == "heavy":
        res = cv2.medianBlur(img, 5)
    else:
        res = cv2.GaussianBlur(img, (5, 5), 0)
    _save_debug(res, f"04_denoised_{strength}")
    return res

def _clahe(img: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    res = clahe.apply(img)
    _save_debug(res, "03_clahe")
    return res

def _threshold_otsu(img: np.ndarray) -> np.ndarray:
    """Otsu global thresholding — best for bimodal histograms."""
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _save_debug(binary, "05_threshold_otsu")
    return binary

def _threshold_adaptive(img: np.ndarray, block: int = 15, c: int = 10) -> np.ndarray:
    """Adaptive thresholding — best for uneven lighting."""
    res = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block, c
    )
    _save_debug(res, "05_threshold_adaptive")
    return res

def _morph_clean(img: np.ndarray) -> np.ndarray:
    """Morphological opening to remove small noise specks."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    res = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    _save_debug(res, "06_morphological_clean")
    return res

def _deskew_rotation(img: np.ndarray) -> np.ndarray:
    """
    Detect 90/180/270 rotations using Tesseract OSD and small skew via Hough lines.
    """
    try:
        # Detect exact 90/180/270 orientation
        osd = pytesseract.image_to_osd(img)
        angle_match = re.search(r'Rotate: (\d+)', osd)
        if angle_match:
            angle = int(angle_match.group(1))
            if angle == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif angle == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            if angle != 0:
                logger.debug(f"OSD rotated image by {angle} degrees")
    except Exception as e:
        logger.warning(f"OSD rotation failed: {e}")

    # Micro-deskew using Hough Transform
    try:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                 minLineLength=100, maxLineGap=10)
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 != x1:
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if abs(angle) < 45:
                        angles.append(angle)

            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) >= 0.5:
                    h, w = img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
                    logger.debug(f"Deskewed micro-angle by {median_angle:.2f}°")
    except Exception as e:
        logger.warning(f"Micro-deskew failed: {e}")
        
    _save_debug(img, "00_deskewed")
    return img

# ── Pipelines ────────────────────────────────────────────────────────────────

def preprocess_printed(img: np.ndarray) -> np.ndarray:
    """Pipeline: resize → gray → CLAHE → denoise → Otsu threshold → morph clean"""
    img = _resize(img)
    gray = _to_gray(img)
    gray = _clahe(gray)
    gray = _denoise(gray, "light")
    binary = _threshold_otsu(gray)
    return _morph_clean(binary)

def preprocess_low_light(img: np.ndarray) -> np.ndarray:
    """Pipeline: resize → gray → aggressive CLAHE → denoise → adaptive threshold"""
    img = _resize(img)
    gray = _to_gray(img)
    gray = _clahe(gray, clip=4.0, tile=4)
    gray = _denoise(gray, "medium")
    return _threshold_adaptive(gray, block=21, c=8)

def preprocess_blurry(img: np.ndarray) -> np.ndarray:
    """Pipeline: resize (larger) → gray → sharpen → CLAHE → Otsu"""
    img = _resize(img, min_width=3000)
    gray = _to_gray(img)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    gray = _clahe(gray)
    return _threshold_otsu(gray)

def preprocess_low_contrast(img: np.ndarray) -> np.ndarray:
    """Pipeline: resize → gray → histogram equalization → adaptive threshold"""
    img = _resize(img)
    gray = _to_gray(img)
    gray = cv2.equalizeHist(gray)
    return _threshold_adaptive(gray)

def preprocess_raw_gray(img: np.ndarray) -> np.ndarray:
    """Pipeline: resize → gray → CLAHE → light denoise (NO thresholding).
    Skipping binarization preserves tonal detail that helps Tesseract
    distinguish visually similar digits like 5 vs 9."""
    img = _resize(img)
    gray = _to_gray(img)
    gray = _clahe(gray)
    gray = _denoise(gray, "light")
    return gray

def preprocess_sharp_gray(img: np.ndarray) -> np.ndarray:
    """Pipeline: resize → gray → sharpen → CLAHE → light denoise (NO thresholding).
    Sharpening emphasises digit edges without the destructive binarization step."""
    img = _resize(img, min_width=3000)
    gray = _to_gray(img)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    gray = _clahe(gray)
    gray = _denoise(gray, "light")
    return gray

def get_preprocessing_candidates(img: np.ndarray, deskew: bool = True) -> Dict[str, np.ndarray]:
    """
    Returns multiple preprocessed versions of the image.
    The calling module can evaluate them by running OCR and picking the best one.
    """
    if deskew:
        img = _deskew_rotation(img)
        
    candidates = {}
    
    # Non-thresholded candidates (preserve tonal detail for digit accuracy)
    try:
        candidates["raw_gray"] = preprocess_raw_gray(img)
    except Exception as e:
        logger.warning(f"Raw gray strategy failed: {e}")

    try:
        candidates["sharp_gray"] = preprocess_sharp_gray(img)
    except Exception as e:
        logger.warning(f"Sharp gray strategy failed: {e}")

    # Thresholded candidates (better for noisy / low-contrast images)
    try:
        candidates["printed"] = preprocess_printed(img)
    except Exception as e:
        logger.warning(f"Printed strategy failed: {e}")
        
    try:
        candidates["low_light"] = preprocess_low_light(img)
    except Exception as e:
        logger.warning(f"Low light strategy failed: {e}")
        
    try:
        candidates["blurry"] = preprocess_blurry(img)
    except Exception as e:
        logger.warning(f"Blurry strategy failed: {e}")
        
    try:
        candidates["low_contrast"] = preprocess_low_contrast(img)
    except Exception as e:
        logger.warning(f"Low contrast strategy failed: {e}")
        
    if not candidates:
        logger.warning("All preprocessing strategies failed. Returning grayscale fallback.")
        candidates["fallback"] = _to_gray(img)
        
    return candidates

def load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img
