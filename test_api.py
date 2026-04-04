"""
test_api.py
SceneSpeak API test scripti.

Kullanım:
    python test_api.py                  # varsayılan: test.jpg kullanır
    python test_api.py resim.jpg        # kendi görüntünü belirt
"""

import base64
import sys
import json
import io
from pathlib import Path
import urllib.request
import urllib.error

# Force UTF-8 output on Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_URL = "http://localhost:8000"


def renkli(metin, kod):
    return f"\033[{kod}m{metin}\033[0m"

yesil   = lambda m: renkli(m, "92")
kirmizi = lambda m: renkli(m, "91")
sari    = lambda m: renkli(m, "93")
kalin   = lambda m: renkli(m, "1")


# ─────────────────────────────────────────────────────────────
# 1. Health check
# ─────────────────────────────────────────────────────────────
def test_health():
    print(kalin("\n── 1. Health Check ──────────────────────────────"))
    try:
        with urllib.request.urlopen(f"{BASE_URL}/api/v1/health") as r:
            data = json.loads(r.read())
        print(yesil("✓ Sunucu çalışıyor"))
        print(f"  detector_loaded : {yesil('true') if data['detector_loaded'] else kirmizi('false')}")
        print(f"  ocr_ready       : {yesil('true') if data['ocr_ready'] else sari('false (Tesseract kurulu değil)')}")
        return True
    except urllib.error.URLError:
        print(kirmizi("✗ Sunucuya bağlanılamadı!"))
        print(sari("  → uvicorn main:app --host 0.0.0.0 --port 8000 --reload"))
        return False


# ─────────────────────────────────────────────────────────────
# 2. Test görüntüsü hazırla
# ─────────────────────────────────────────────────────────────
def goruntu_hazirla(dosya_yolu: str) -> str:
    path = Path(dosya_yolu)

    if not path.exists():
        print(sari(f"  '{dosya_yolu}' bulunamadı, örnek görüntü indiriliyor…"))
        ornek_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"
        urllib.request.urlretrieve(ornek_url, dosya_yolu)
        print(yesil(f"  ✓ Örnek görüntü indirildi: {dosya_yolu}"))

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    print(yesil(f"  ✓ Görüntü yüklendi: {path.name}  ({path.stat().st_size // 1024} KB)"))
    return b64


# ─────────────────────────────────────────────────────────────
# 3. Detect endpoint testi
# ─────────────────────────────────────────────────────────────
def test_detect(b64_image: str):
    print(kalin("\n── 2. Detect Testi ──────────────────────────────"))

    payload = json.dumps({"image": b64_image}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/detect",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        hata = json.loads(e.read())
        print(kirmizi(f"✗ HTTP {e.code}: {hata.get('detail', e.reason)}"))
        return

    # Tespit edilen nesneler
    objects = data.get("objects", [])
    print(yesil(f"✓ {len(objects)} nesne tespit edildi:"))
    for obj in objects:
        print(f"   • {obj['label']:<20} güven: {obj['confidence']:.0%}  "
              f"konum: x={obj['x']} y={obj['y']} w={obj['w']} h={obj['h']}")

    if not objects:
        print(sari("   (Hiç nesne bulunamadı — görüntüyü değiştirip tekrar dene)"))

    # OCR metinleri
    texts = data.get("texts", [])
    print(yesil(f"\n✓ OCR — {len(texts)} metin satırı:"))
    for t in texts:
        print(f"   \"{t}\"")
    if not texts:
        print(sari("   (Metin bulunamadı)"))

    # Guidance
    guidance = data.get("guidance", "")
    print(kalin(f"\n🔊 Sesli rehberlik: ") + yesil(f'"{guidance}"'))


# ─────────────────────────────────────────────────────────────
# 4. Hatalı istek testi
# ─────────────────────────────────────────────────────────────
def test_hatali_istek():
    print(kalin("\n── 3. Hatalı İstek Testi (400 bekleniyor) ──────"))
    payload = json.dumps({"image": "bu_gecersiz_base64!!"}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/detect",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req)
        print(kirmizi("✗ 400 hatası bekleniyor ama gelmedi"))
    except urllib.error.HTTPError as e:
        if e.code == 400:
            print(yesil(f"✓ Doğru — sunucu 400 Bad Request döndürdü"))
        else:
            print(sari(f"  HTTP {e.code} döndü (400 bekleniyor)"))


# ─────────────────────────────────────────────────────────────
# Ana akış
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    goruntu_dosyasi = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

    print(kalin("═══════════════════════════════════════════════"))
    print(kalin("   SceneSpeak API Test"))
    print(kalin("═══════════════════════════════════════════════"))

    if not test_health():
        sys.exit(1)

    print(kalin("\n── Görüntü Hazırlanıyor ─────────────────────────"))
    b64 = goruntu_hazirla(goruntu_dosyasi)

    test_detect(b64)
    test_hatali_istek()

    print(kalin("\n═══════════════════════════════════════════════"))
    print(yesil("  Tüm testler tamamlandı."))
    print(kalin("═══════════════════════════════════════════════\n"))
