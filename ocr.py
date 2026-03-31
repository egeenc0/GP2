"""
ocr.py
Tesseract 5 OCR wrapper supporting Turkish and English.

Tesseract must be installed separately (see README.md).
If pytesseract cannot locate Tesseract, OCR silently returns an empty list
so the rest of the pipeline keeps working.
"""

import logging
import re

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

# Tesseract config: LSTM engine (--oem 1), treat image as sparse text (--psm 11)
# PSM 11 works well for signs, labels, and mixed layouts in real-world photos.
_TESS_CONFIG = "--oem 1 --psm 11"
_TESS_LANG = "tur+eng"

# Minimum character length to keep a recognised text token
_MIN_TOKEN_LEN = 2


def _preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """Convert to greyscale, upscale if small, then apply adaptive threshold.

    These steps significantly improve Tesseract accuracy on low-contrast
    or small-font text captured by a phone camera.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Upscale if the image is smaller than 600 px wide — Tesseract prefers >= 300 dpi
    h, w = gray.shape
    if w < 600:
        scale = 600 / w
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Denoise before thresholding
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold copes with uneven illumination
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return binary


def _clean_tokens(raw: str) -> list[str]:
    """Split raw Tesseract output into meaningful tokens, discarding noise."""
    tokens: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        # Drop lines that are purely punctuation / whitespace / single characters
        cleaned = re.sub(r"[^\w\s]", "", line, flags=re.UNICODE).strip()
        if len(cleaned) >= _MIN_TOKEN_LEN:
            tokens.append(line)  # keep original punctuation for readability
    return tokens


class OCRReader:
    """Thin wrapper around pytesseract. Call .load() at startup to validate install."""

    def __init__(self) -> None:
        self._available = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Check that Tesseract is accessible; log a warning if not."""
        try:
            version = pytesseract.get_tesseract_version()
            logger.info("Tesseract %s detected.", version)
            self._available = True
        except pytesseract.TesseractNotFoundError:
            logger.warning(
                "Tesseract executable not found. OCR will be disabled. "
                "Install Tesseract 5 and add it to PATH (see README.md)."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tesseract probe failed: %s. OCR disabled.", exc)

    def is_ready(self) -> bool:
        return self._available

    def read(self, img: np.ndarray) -> list[str]:
        """Run OCR on a BGR image and return a list of cleaned text lines.

        Returns an empty list if Tesseract is unavailable or no text is found.
        """
        if not self._available:
            return []

        try:
            processed = _preprocess_for_ocr(img)
            raw = pytesseract.image_to_string(processed, lang=_TESS_LANG, config=_TESS_CONFIG)
            return _clean_tokens(raw)
        except pytesseract.TesseractNotFoundError:
            self._available = False
            logger.error("Tesseract disappeared from PATH during runtime.")
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR error: %s", exc)
            return []
