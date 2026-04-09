"""
main.py
SceneSpeak FastAPI application.

Endpoints
---------
GET  /api/v1/health    — liveness check, reports model status
POST /api/v1/detect    — accepts a base64 JPEG, returns detected objects,
                         OCR text, and a spoken guidance string
POST /api/v1/speak     — accepts {"text": "..."} and returns MP3 audio (edge-tts)
POST /api/v1/detect-speak — detect + TTS in a single call, returns MP3 audio
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from detector import ObjectDetector
from guidance import generate_guidance
from ocr import OCRReader
from preprocessor import TARGET_HEIGHT, TARGET_WIDTH, decode_base64_image, preprocess
from tts import DEFAULT_VOICE, FALLBACK_VOICE, synthesise

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


class SpeakRequest(BaseModel):
    text: str = Field(..., description="Seslendirilecek metin.")
    voice: str = Field(
        default=DEFAULT_VOICE,
        description=f"edge-tts ses adı. Varsayılan: {DEFAULT_VOICE}. "
                    f"Alternatif: {FALLBACK_VOICE}",
    )


class DetectSpeakRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded JPEG image.")
    voice: str = Field(default=DEFAULT_VOICE, description="edge-tts ses adı.")


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


@app.post(
    "/api/v1/speak",
    summary="Metni MP3 ses dosyasına dönüştür (TTS)",
    response_class=Response,
    responses={
        200: {"content": {"audio/mpeg": {}}, "description": "MP3 ses verisi"},
        503: {"description": "TTS hazır değil (edge-tts kurulu değil veya ağ yok)"},
    },
)
async def speak(request: SpeakRequest) -> Response:
    """Verilen metni Türkçe Neural TTS ile MP3 olarak döndürür.

    Flutter / Android istemcisi bu ses akışını doğrudan oynatabilir.
    ```json
    { "text": "Saat üç yönünde kişi var.", "voice": "tr-TR-EmelNeural" }
    ```
    """
    audio = await synthesise(request.text, request.voice)
    if not audio:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS üretilemedi. İnternet bağlantısını ve edge-tts kurulumunu kontrol edin.",
        )
    return Response(content=audio, media_type="audio/mpeg")


@app.post(
    "/api/v1/detect-speak",
    summary="Görüntüyü analiz et ve sesli rehberlik MP3'ünü döndür",
    response_class=Response,
    responses={
        200: {"content": {"audio/mpeg": {}}, "description": "MP3 rehberlik sesi"},
    },
)
async def detect_speak(request: DetectSpeakRequest) -> Response:
    """Tek istekle: görüntü analizi + TTS → MP3 ses.

    Flutter kameradan aldığı kareyi gönderir, doğrudan ses oynatır.
    ```json
    { "image": "<base64 JPEG>", "voice": "tr-TR-EmelNeural" }
    ```
    """
    img = decode_base64_image(request.image)
    if img is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Geçersiz base64 JPEG.")

    processed = preprocess(img)

    if not detector.is_loaded():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Nesne dedektörü henüz hazır değil.")

    objects = detector.detect(processed)
    texts = ocr_reader.read(processed)
    guidance_text = generate_guidance(objects, texts, TARGET_WIDTH, TARGET_HEIGHT)

    logger.info("detect-speak() → %d objects | guidance: %r", len(objects), guidance_text)

    audio = await synthesise(guidance_text, request.voice)
    if not audio:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="TTS üretilemedi.")
    return Response(content=audio, media_type="audio/mpeg")
