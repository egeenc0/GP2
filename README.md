# SceneSpeak Backend

Python + FastAPI backend for the **SceneSpeak** assistive mobile app.  
It receives a base64-encoded JPEG from a Flutter client, runs MobileNet-SSD object detection and Tesseract OCR, and returns structured JSON with a spoken guidance string.

---

## Project Structure

```
scenespeak-backend/
├── main.py           FastAPI app — endpoints and startup logic
├── detector.py       MobileNet SSD v1 (TFLite) object detection
├── ocr.py            Tesseract 5 OCR (Turkish + English)
├── preprocessor.py   OpenCV image decoding and CLAHE preprocessing
├── guidance.py       Priority-based guidance text generator
├── requirements.txt
├── models/           Auto-created; .tflite model downloaded here on first run
└── README.md
```

---

## Prerequisites

### 1. Python 3.11

```bash
python --version   # should print Python 3.11.x
```

### 2. Tesseract 5

#### Windows
1. Download the installer from <https://github.com/UB-Mannheim/tesseract/wiki>  
   (choose the latest **5.x** version).
2. Run the installer.  During installation tick **Additional language data → Turkish (tur)**.
3. Add the install directory (e.g. `C:\Program Files\Tesseract-OCR`) to your `PATH`  
   **or** set the `TESSDATA_PREFIX` environment variable.
4. Verify:
   ```powershell
   tesseract --version
   ```

#### Ubuntu / Debian
```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-tur
```

#### macOS
```bash
brew install tesseract
# For Turkish language data:
brew install tesseract-lang
```

---

## Setup

```bash
# Clone / open the project folder
cd scenespeak-backend

# (Recommended) create a virtual environment
python -m venv .venv

# Activate it
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

> **Note:** `tensorflow` is large (~500 MB). On slower connections this may take a while.  
> If you only need TFLite inference you can replace `tensorflow` with the lighter  
> `tflite-runtime` package and adjust the import in `detector.py`.

---

## Running the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

On first startup the backend automatically downloads the MobileNet SSD TFLite model  
(~6 MB) into `./models/`. Subsequent startups load from disk.

The API will be available at:  
- Base URL: `http://localhost:8000`  
- Interactive docs: `http://localhost:8000/docs`

---

## API Reference

### `GET /api/v1/health`

Returns server and model readiness status.

**Response:**
```json
{
  "status": "ok",
  "detector_loaded": true,
  "ocr_ready": true
}
```

---

### `POST /api/v1/detect`

Analyse a camera frame and return detected objects, text, and spoken guidance.

**Request body (JSON):**
```json
{
  "image": "<base64-encoded JPEG string>"
}
```

Both raw base64 and data-URI format (`data:image/jpeg;base64,...`) are accepted.

**Response:**
```json
{
  "objects": [
    {
      "label": "person",
      "confidence": 0.92,
      "x": 120,
      "y": 80,
      "w": 200,
      "h": 310
    }
  ],
  "texts": ["EXIT", "Çıkış"],
  "guidance": "Person ahead. Sign reads: EXIT; Çıkış."
}
```

| Field | Description |
|---|---|
| `objects` | List of detected objects with bounding boxes in pixel coordinates |
| `texts` | OCR-extracted text lines from the image |
| `guidance` | Short spoken string ready for text-to-speech |

Guidance priority:  
1. **Obstacles near centre** (person, car, chair, …) → `"Person ahead."`  
2. **Other detected objects** → `"Detected: bottle, cup."`  
3. **OCR text** → `"Sign reads: EXIT."` (appended to the above)  
4. **Fallback** → `"No obstacles detected."`

---

## Flutter Integration (quick start)

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

Future<Map<String, dynamic>> analyseFrame(Uint8List jpegBytes) async {
  final response = await http.post(
    Uri.parse('http://<YOUR_SERVER_IP>:8000/api/v1/detect'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({'image': base64Encode(jpegBytes)}),
  );
  return jsonDecode(response.body) as Map<String, dynamic>;
}
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `TesseractNotFoundError` | Add Tesseract to `PATH` or set `pytesseract.pytesseract.tesseract_cmd` in `ocr.py` |
| Model download fails | Download manually from the URL in `detector.py` and place as `models/ssd_mobilenet_v1_metadata.tflite` |
| `tensorflow` import error on Windows ARM | Use `tflite-runtime` instead (`pip install tflite-runtime`) |
| Low OCR accuracy | Ensure Tesseract `tur` data is installed; try better lighting |

---

## Notes

- No authentication, no database — prototype only.
- CORS is open (`*`) for development. Restrict origins before any public deployment.
- Confidence threshold defaults to **0.40**; adjust `DEFAULT_CONFIDENCE_THRESHOLD` in `detector.py`.
