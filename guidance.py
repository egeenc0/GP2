"""
guidance.py
Priority-based spoken guidance generator for SceneSpeak.

Priority order (highest → lowest):
  1. Obstacles near the centre of the frame (person, car, chair, …)
  2. Non-obstacle objects detected anywhere in the frame
  3. OCR-detected text / signs
  4. Default "no obstacles detected" fallback

The centre region is a configurable rectangular zone.  Objects whose
bounding-box centre falls inside it are treated as immediate obstacles.
"""

# Classes that constitute navigation hazards
OBSTACLE_CLASSES: frozenset[str] = frozenset(
    {
        "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
        "chair", "couch", "bed", "dining table", "dog", "cat", "horse",
        "traffic light", "stop sign", "fire hydrant",
    }
)

# Fraction of frame width/height that defines the central danger zone
# 0.35 → centre 35 % on each axis ≈ the middle ~49 % area
_CENTRE_ZONE_X = 0.35
_CENTRE_ZONE_Y = 0.35

# Maximum objects / text tokens to mention in one guidance string
_MAX_OBSTACLES = 3
_MAX_GENERAL = 3
_MAX_TEXTS = 2


def _is_near_centre(obj: dict, img_w: int, img_h: int) -> bool:
    """Return True if the bounding-box centre lies in the central danger zone."""
    cx = (obj["x"] + obj["w"] / 2) / img_w
    cy = (obj["y"] + obj["h"] / 2) / img_h
    return abs(cx - 0.5) <= _CENTRE_ZONE_X and abs(cy - 0.5) <= _CENTRE_ZONE_Y


def _deduplicate(labels: list[str]) -> list[str]:
    """Preserve order while removing duplicate labels."""
    seen: set[str] = set()
    out: list[str] = []
    for lbl in labels:
        if lbl not in seen:
            seen.add(lbl)
            out.append(lbl)
    return out


def generate_guidance(
    objects: list[dict],
    texts: list[str],
    img_w: int = 640,
    img_h: int = 480,
) -> str:
    """Build a short spoken guidance string from detection and OCR results.

    Parameters
    ----------
    objects:
        List of dicts with keys label, confidence, x, y, w, h.
    texts:
        List of OCR-extracted text lines.
    img_w, img_h:
        Dimensions of the preprocessed frame (used for centre-zone maths).

    Returns
    -------
    A human-readable guidance sentence suitable for text-to-speech.
    """
    centre_obstacles: list[dict] = []
    general_objects: list[dict] = []

    for obj in objects:
        if obj["label"] in OBSTACLE_CLASSES and _is_near_centre(obj, img_w, img_h):
            centre_obstacles.append(obj)
        else:
            general_objects.append(obj)

    # Sort each group by confidence descending
    centre_obstacles.sort(key=lambda o: o["confidence"], reverse=True)
    general_objects.sort(key=lambda o: o["confidence"], reverse=True)

    parts: list[str] = []

    # --- Priority 1: central obstacles ---
    if centre_obstacles:
        top = centre_obstacles[0]
        parts.append(f"{top['label'].capitalize()} ahead.")

        extras = _deduplicate(
            [o["label"] for o in centre_obstacles[1:_MAX_OBSTACLES]]
        )
        if extras:
            parts.append(f"Also nearby: {', '.join(extras)}.")

    # --- Priority 2: general objects (non-central or non-obstacle class) ---
    elif general_objects:
        labels = _deduplicate([o["label"] for o in general_objects[:_MAX_GENERAL]])
        parts.append(f"Detected: {', '.join(labels)}.")

    # --- Priority 3: OCR text ---
    clean_texts = [t.strip() for t in texts if len(t.strip()) >= 2]
    if clean_texts:
        snippet = "; ".join(clean_texts[:_MAX_TEXTS])
        parts.append(f"Sign reads: {snippet}.")

    # --- Fallback ---
    if not parts:
        parts.append("No obstacles detected.")

    return " ".join(parts)
