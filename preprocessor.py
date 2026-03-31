"""
preprocessor.py
Handles base64 image decoding and OpenCV preprocessing (resize + CLAHE).
"""

import base64
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Target display resolution for detection and OCR
TARGET_WIDTH = 640
TARGET_HEIGHT = 480


def decode_base64_image(b64_string: str) -> Optional[np.ndarray]:
    """Decode a base64-encoded JPEG/PNG string into a BGR NumPy array.

    Accepts both raw base64 and data-URI format (data:image/jpeg;base64,...).
    Returns None if decoding fails.
    """
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]

        padding = 4 - len(b64_string) % 4
        if padding != 4:
            b64_string += "=" * padding

        img_bytes = base64.b64decode(b64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            logger.warning("cv2.imdecode returned None — invalid image bytes.")
        return img
    except Exception as exc:
        logger.error("Failed to decode base64 image: %s", exc)
        return None


def preprocess(img: np.ndarray) -> np.ndarray:
    """Resize to TARGET_WIDTH×TARGET_HEIGHT and apply CLAHE brightness normalisation.

    CLAHE (Contrast Limited Adaptive Histogram Equalisation) improves detection
    and OCR accuracy under uneven lighting conditions common in mobile captures.
    """
    resized = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    # Work in LAB colour space so only the luminance channel is equalised
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)

    lab_eq = cv2.merge((l_eq, a_channel, b_channel))
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return enhanced
