"""
main.py
SceneSpeak FastAPI application.

Endpoints
---------
GET  /api/v1/health   — liveness check, reports model status
POST /api/v1/detect   — accepts a base64 JPEG, returns detected objects,
                        OCR text, and a spoken guidance string
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from detector import ObjectDetector
from guidance import generate_guidance
from ocr import OCRReader
from preprocessor import TARGET_HEIGHT, TARGET_WIDTH, decode_base64_image, preprocess

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model instances (loaded once at startup)
# ---------------------------------------------------------------------------
detector = ObjectDetector()
ocr_reader = OCRReader()


# ---------------------------------------------------------------------------
# Lifespan — load heavy models before serving any requests
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    logger.info("SceneSpeak startup: loading models…")
    detector.load()
    ocr_reader.load()
    logger.info("All models ready. Serving requests.")
    yield
    logger.info("SceneSpeak shutdown.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SceneSpeak API",
    description="Assistive vision backend — object detection + OCR + spoken guidance.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class DetectRequest(BaseModel):
    image: str = Field(
        ...,
        description="Base64-encoded JPEG image (raw or data-URI format).",
    )


class DetectedObject(BaseModel):
    label: str
    confidence: float
    x: int
    y: int
    w: int
    h: int


class DetectResponse(BaseModel):
    objects: list[DetectedObject]
    texts: list[str]
    guidance: str
    frame_width: int = Field(
        default=TARGET_WIDTH,
        description="Bounding box koordinatlarının referans genişliği (piksel).",
    )
    frame_height: int = Field(
        default=TARGET_HEIGHT,
        description="Bounding box koordinatlarının referans yüksekliği (piksel).",
    )


# ---------------------------------------------------------------------------
# Static demo UI (same origin → no CORS issues)
# ---------------------------------------------------------------------------
_STATIC_DIR = Path(__file__).resolve().parent / "static"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def index_page() -> FileResponse:
    """Simple browser UI to upload a photo and call /api/v1/detect."""
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/api/v1/health", summary="Health check")
async def health() -> dict:
    """Return service status and model readiness flags."""
    return {
        "status": "ok",
        "detector_loaded": detector.is_loaded(),
        "ocr_ready": ocr_reader.is_ready(),
    }


@app.post(
    "/api/v1/detect",
    response_model=DetectResponse,
    summary="Analyse a camera frame",
    status_code=status.HTTP_200_OK,
)
async def detect(request: DetectRequest) -> DetectResponse:
    """Decode the base64 image, run object detection and OCR, return guidance.

    The Flutter client should POST JSON:
    ```json
    { "image": "<base64-encoded JPEG>" }
    ```
    """
    # 1. Decode
    img = decode_base64_image(request.image)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image. Ensure the payload is a valid base64 JPEG.",
        )

    # 2. Preprocess
    processed = preprocess(img)

    # 3. Detect objects
    if not detector.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Object detector is not ready yet. Please retry in a moment.",
        )
    objects = detector.detect(processed)

    # 4. OCR
    texts = ocr_reader.read(processed)

    # 5. Generate guidance
    guidance_text = generate_guidance(objects, texts, TARGET_WIDTH, TARGET_HEIGHT)

    logger.info(
        "detect() → %d objects, %d text lines | guidance: %r",
        len(objects),
        len(texts),
        guidance_text,
    )

    return DetectResponse(
        objects=[DetectedObject(**obj) for obj in objects],
        texts=texts,
        guidance=guidance_text,
        frame_width=TARGET_WIDTH,
        frame_height=TARGET_HEIGHT,
    )
