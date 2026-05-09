"""
ocr/ocr_engine.py
-----------------
Tesseract OCR engine wrapper with fallback to EasyOCR.
Evaluates multiple preprocessing candidates dynamically for maximum accuracy.
"""

import logging
import platform
import os
import shutil
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np
import pytesseract

from ocr.preprocessing import get_preprocessing_candidates, load_image
from ocr.postprocessing import merge_boxes, extract_text_from_merged

logger = logging.getLogger(__name__)


# ── Tesseract path auto-detection ─────────────────────────────────────────────

def _find_tesseract() -> Optional[str]:
    """Auto-detect Tesseract executable path across platforms."""
    system = platform.system()
    candidates = []

    if system == "Windows":
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\aryan_9nptyqz\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        ]
    elif system == "Darwin":
        candidates = [
            "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract",
        ]
    else:  # Linux
        candidates = [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
        ]

    for path in candidates:
        if os.path.isfile(path):
            logger.debug(f"Found Tesseract at: {path}")
            return path

    found = shutil.which("tesseract")
    if found:
        return found

    logger.warning("Tesseract not found. Please ensure it is installed.")
    return None

def configure_tesseract(custom_path: Optional[str] = None) -> None:
    """Configure pytesseract with the correct Tesseract binary path."""
    path = custom_path or _find_tesseract()
    if path:
        pytesseract.pytesseract.tesseract_cmd = path


# ── OCREngine Class ───────────────────────────────────────────────────────────

class OCRError(Exception):
    pass

class OCREngine:
    def __init__(self, lang: str = "eng", use_easyocr_fallback: bool = True):
        self.lang = lang
        self.use_easyocr_fallback = use_easyocr_fallback
        self._easyocr_reader = None
        configure_tesseract()
        
    def _get_easyocr(self):
        """Lazy-load the EasyOCR reader if it is needed."""
        if self._easyocr_reader is None:
            try:
                import easyocr
                # Init with gpu=True if available. It gracefully falls back to CPU.
                self._easyocr_reader = easyocr.Reader([self.lang], gpu=True)
            except ImportError:
                logger.warning("EasyOCR not installed. Fallback is disabled.")
                self.use_easyocr_fallback = False
        return self._easyocr_reader

    def get_best_preprocessing(self, img: np.ndarray, config: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Runs a fast Tesseract pass on all candidates and returns the best image and its OCR data."""
        candidates = get_preprocessing_candidates(img)
        best_score = -1
        best_img = None
        best_data = []

        for name, candidate_img in candidates.items():
            try:
                # Use image_to_data to get bounding boxes and confidences
                data = pytesseract.image_to_data(
                    candidate_img, lang=self.lang, config=config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Filter valid words and sum confidence
                confs = []
                valid_words_data = []
                for i, word in enumerate(data["text"]):
                    conf = int(data["conf"][i])
                    # Require minimum conf to count as "text" for evaluation
                    if conf > 10 and word.strip():
                        confs.append(conf)
                        valid_words_data.append({
                            "text": word,
                            "conf": conf,
                            "left": data["left"][i],
                            "top": data["top"][i],
                            "width": data["width"][i],
                            "height": data["height"][i],
                            "block_num": data["block_num"][i],
                            "par_num": data["par_num"][i],
                            "line_num": data["line_num"][i],
                        })
                
                avg_conf = sum(confs) / len(confs) if confs else 0
                logger.debug(f"Preprocessing candidate '{name}' scored avg conf {avg_conf:.2f}")

                if avg_conf > best_score:
                    best_score = avg_conf
                    best_img = candidate_img
                    best_data = valid_words_data
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate candidate '{name}': {e}")
                
        if best_img is None:
            raise OCRError("All preprocessing evaluations failed.")
            
        return best_img, best_data

    def extract_data(self, img: np.ndarray, min_confidence: int = 30) -> List[Dict[str, Any]]:
        """
        Dynamically select the best config and preprocessing pipeline, 
        yielding the most structured and confident bounding boxes.
        """
        configs_to_try = [
            "--psm 6 --oem 3",   # Uniform block of text
            "--psm 11 --oem 3"   # Sparse text layout
        ]
        
        best_overall_score = -1
        best_overall_data = []
        
        for config in configs_to_try:
            try:
                # Determine the best image transformation for this PSM
                _, raw_data = self.get_best_preprocessing(img, config)
                
                # Hard filter the result based on requested minimal confidence
                filtered_data = [w for w in raw_data if w['conf'] >= min_confidence]
                
                if not filtered_data:
                    continue
                    
                # Weight by both confidence and the amount of recognized text
                # We cap count at 100 so length doesn't completely overpower average confidence
                avg_conf = sum(w['conf'] for w in filtered_data) / len(filtered_data)
                score = avg_conf * min(len(filtered_data), 100) 
                
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_data = filtered_data
                    
            except Exception as e:
                logger.error(f"OCR evaluation config {config} failed: {e}")
                
        # EasyOCR Fallback Check
        if not best_overall_data and self.use_easyocr_fallback:
            reader = self._get_easyocr()
            if reader:
                logger.info("Tesseract failed. Invoking EasyOCR fallback...")
                try:
                    candidates = get_preprocessing_candidates(img)
                    easy_img = candidates.get("printed", img) # EasyOCR usually likes printed pipeline
                    
                    results = reader.readtext(easy_img)
                    for (bbox, text, prob) in results:
                        conf = int(prob * 100)
                        if conf >= min_confidence:
                            best_overall_data.append({
                                "text": text,
                                "conf": conf,
                                "left": int(bbox[0][0]),
                                "top": int(bbox[0][1]),
                                "width": int(bbox[1][0] - bbox[0][0]),
                                "height": int(bbox[2][1] - bbox[0][1]),
                            })
                except Exception as e:
                    logger.error(f"EasyOCR fallback failed: {e}")

        # Final Post-processing: Merge aligned bounding boxes into sentences/lines
        merged_lines = merge_boxes(best_overall_data)
        return merged_lines

    def ocr_text(self, img: np.ndarray) -> str:
        """Helper to get a clean extracted string block."""
        data = self.extract_data(img)
        return extract_text_from_merged(data)

    def all_candidates_text(self, img: np.ndarray, min_confidence: int = 30) -> list:
        """
        Run OCR on EVERY preprocessing candidate with MULTIPLE Tesseract
        configs and return a list of text strings.  Downstream code can
        extract CBC values from each and take a majority vote to neutralise
        digit misreads (e.g. Tesseract reading '5' as '9').
        """
        from ocr.postprocessing import merge_boxes as _merge, extract_text_from_merged as _etfm
        candidates = get_preprocessing_candidates(img)
        configs = [
            "--psm 6 --oem 3",   # Uniform block of text
            "--psm 4 --oem 3",   # Single column of variable-size text
            "--psm 3 --oem 3",   # Fully automatic page segmentation
        ]
        texts = []

        for name, candidate_img in candidates.items():
            for config in configs:
                try:
                    data = pytesseract.image_to_data(
                        candidate_img, lang=self.lang, config=config,
                        output_type=pytesseract.Output.DICT,
                    )
                    valid = []
                    for i, word in enumerate(data["text"]):
                        conf = int(data["conf"][i])
                        if conf >= min_confidence and word.strip():
                            valid.append({
                                "text": word, "conf": conf,
                                "left": data["left"][i], "top": data["top"][i],
                                "width": data["width"][i], "height": data["height"][i],
                                "block_num": data["block_num"][i],
                                "par_num": data["par_num"][i],
                                "line_num": data["line_num"][i],
                            })
                    if valid:
                        merged = _merge(valid)
                        texts.append(_etfm(merged))
                        logger.debug(
                            f"all_candidates_text: '{name}' + '{config}' → {len(merged)} lines"
                        )
                except Exception as e:
                    logger.warning(f"all_candidates_text: '{name}' + '{config}' failed: {e}")

        return texts

    def process_file(self, path: str, output: str = "text") -> Any:
        try:
            img = load_image(path)
            if output == "text":
                return self.ocr_text(img)
            elif output == "json":
                return self.extract_data(img)
            else:
                raise ValueError("Only 'text' and 'json' outputs are supported in the new pipeline.")
        except Exception as e:
            logger.error(f"Failed to process file {path}: {e}")
            return ""


# ── Backward Compatibility ────────────────────────────────────────────────────

_default_engine = OCREngine()

def ocr_text(img: np.ndarray, **kwargs) -> str:
    return _default_engine.ocr_text(img)

def ocr_json(img: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
    # To keep identical signature kwargs compatibility
    min_conf = kwargs.get("min_confidence", 30)
    return _default_engine.extract_data(img, min_confidence=min_conf)

def ocr_file(path: str, **kwargs) -> Any:
    return _default_engine.process_file(path, kwargs.get("output", "text"))
