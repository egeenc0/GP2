"""
detector.py
MobileNet SSD v1 object detection via TensorFlow Lite.

The .tflite model is downloaded automatically on first startup into ./models/.
Model source: TensorFlow Lite Task Library (COCO 90-class, uint8 quant).
"""

import logging
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite"
    "/task_library/object_detection/android/"
    "lite-model_ssd_mobilenet_v1_1_metadata_2.tflite"
)
MODEL_PATH = Path("models/ssd_mobilenet_v1_metadata.tflite")

# COCO label map — index 0 is background; indices follow the standard 1-based COCO IDs.
# Gaps in the original COCO numbering are filled with empty strings.
COCO_LABELS: dict[int, str] = {
    0: "",
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 12: "", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    26: "", 27: "backpack", 28: "umbrella", 29: "", 30: "",
    31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis",
    36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove",
    41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle", 45: "",
    46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon",
    51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange",
    56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut",
    61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed",
    66: "", 67: "dining table", 68: "", 69: "", 70: "toilet",
    71: "", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster",
    81: "sink", 82: "refrigerator", 83: "", 84: "book", 85: "clock",
    86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}

DEFAULT_CONFIDENCE_THRESHOLD = 0.40


def _download_model() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MobileNet SSD model from TensorFlow servers…")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    logger.info("Model saved to %s (%.1f MB)", MODEL_PATH, MODEL_PATH.stat().st_size / 1e6)


class ObjectDetector:
    """Wraps a TFLite SSD interpreter. Call .load() once at startup."""

    def __init__(self) -> None:
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._input_h: int = 300
        self._input_w: int = 300

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Download model if absent, then load TFLite interpreter."""
        if not MODEL_PATH.exists():
            _download_model()

        try:
            import tensorflow as tf  # noqa: PLC0415
            self._interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
        except ImportError:
            # Fallback: tflite-runtime only
            from tflite_runtime.interpreter import Interpreter  # noqa: PLC0415
            self._interpreter = Interpreter(model_path=str(MODEL_PATH))

        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        shape = self._input_details[0]["shape"]  # [1, H, W, 3]
        self._input_h = int(shape[1])
        self._input_w = int(shape[2])
        logger.info(
            "TFLite interpreter ready. Input shape: %s, dtype: %s",
            shape,
            self._input_details[0]["dtype"],
        )

    def is_loaded(self) -> bool:
        return self._interpreter is not None

    def detect(
        self, img: np.ndarray, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ) -> list[dict]:
        """Run inference on a BGR image and return detected objects.

        Returns a list of dicts with keys:
            label (str), confidence (float), x (int), y (int), w (int), h (int)
        Coordinates are in pixels relative to the input image dimensions.
        """
        if self._interpreter is None:
            raise RuntimeError("ObjectDetector not loaded. Call .load() first.")

        img_h, img_w = img.shape[:2]

        # Prepare model input
        model_input = cv2.resize(img, (self._input_w, self._input_h))
        model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)

        input_dtype = self._input_details[0]["dtype"]
        if input_dtype == np.uint8:
            tensor = np.expand_dims(model_input.astype(np.uint8), axis=0)
        else:
            tensor = np.expand_dims(model_input.astype(np.float32) / 255.0, axis=0)

        self._interpreter.set_tensor(self._input_details[0]["index"], tensor)
        self._interpreter.invoke()

        # The metadata model outputs are ordered:
        # [0] boxes [1, N, 4]  — normalised [y_min, x_min, y_max, x_max]
        # [1] classes [1, N]   — float, cast to int for label lookup
        # [2] scores [1, N]    — confidence in [0, 1]
        # [3] count  [1]       — number of valid detections
        boxes   = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
        classes = self._interpreter.get_tensor(self._output_details[1]["index"])[0]
        scores  = self._interpreter.get_tensor(self._output_details[2]["index"])[0]

        results: list[dict] = []
        for i, score in enumerate(scores):
            if score < threshold:
                continue

            class_id = int(classes[i]) + 1  # model outputs 0-based; COCO map is 1-based
            label = COCO_LABELS.get(class_id, "")
            if not label:
                continue

            y_min, x_min, y_max, x_max = boxes[i]
            x = max(0, int(x_min * img_w))
            y = max(0, int(y_min * img_h))
            w = int((x_max - x_min) * img_w)
            h = int((y_max - y_min) * img_h)

            results.append(
                {
                    "label": label,
                    "confidence": round(float(score), 4),
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
            )

        return results
