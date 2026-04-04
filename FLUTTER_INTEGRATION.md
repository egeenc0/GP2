# SceneSpeak — Flutter Integration Guide

Backend URL: `http://<SUNUCU_IP>:8000`

---

## pubspec.yaml bağımlılıkları

```yaml
dependencies:
  http: ^1.2.1
  camera: ^0.10.5+9
  flutter_tts: ^4.0.2
```

---

## 1. Health Check

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<bool> isServerReady() async {
  final res = await http.get(Uri.parse('http://SUNUCU_IP:8000/api/v1/health'));
  final data = jsonDecode(res.body);
  return data['status'] == 'ok' && data['detector_loaded'] == true;
}
```

---

## 2. Görüntü Gönder → Rehberlik Al

```dart
import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;

Future<Map<String, dynamic>> analyze(Uint8List jpegBytes) async {
  final response = await http.post(
    Uri.parse('http://SUNUCU_IP:8000/api/v1/detect'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({'image': base64Encode(jpegBytes)}),
  );

  if (response.statusCode == 200) {
    return jsonDecode(response.body) as Map<String, dynamic>;
  }
  throw Exception('Hata: ${response.statusCode}');
}
```

---

## 3. Kameradan Fotoğraf Çek ve Analiz Et

```dart
import 'package:camera/camera.dart';
import 'package:flutter_tts/flutter_tts.dart';

final FlutterTts tts = FlutterTts();
late CameraController cameraController;

Future<void> captureAndAnalyze() async {
  // Fotoğraf çek
  final XFile photo = await cameraController.takePicture();
  final Uint8List bytes = await photo.readAsBytes();

  // Sunucuya gönder
  final result = await analyze(bytes);

  // Sesli rehberliği oku
  final String guidance = result['guidance'];
  await tts.speak(guidance);
}
```

---

## 4. API Yanıtı — JSON Yapısı

```json
{
  "objects": [
    {
      "label": "person",
      "confidence": 0.91,
      "x": 120,
      "y": 45,
      "w": 180,
      "h": 310
    }
  ],
  "texts": ["EXIT", "Çıkış"],
  "guidance": "Person ahead. Sign reads: EXIT."
}
```

| Alan | Tip | Açıklama |
|---|---|---|
| `objects` | list | Tespit edilen nesneler |
| `objects[].label` | string | Nesne adı (İngilizce, 80 COCO sınıfı) |
| `objects[].confidence` | float | 0.0 – 1.0 güven skoru |
| `objects[].x/y/w/h` | int | Piksel cinsinden bounding box |
| `texts` | list | OCR ile okunan metin satırları |
| `guidance` | string | TTS'e direkt verilecek sesli rehberlik |

---

## 5. Dart Model Sınıfları

```dart
class DetectedObject {
  final String label;
  final double confidence;
  final int x, y, w, h;

  DetectedObject.fromJson(Map<String, dynamic> j)
      : label = j['label'],
        confidence = (j['confidence'] as num).toDouble(),
        x = j['x'], y = j['y'], w = j['w'], h = j['h'];
}

class DetectionResult {
  final List<DetectedObject> objects;
  final List<String> texts;
  final String guidance;

  DetectionResult.fromJson(Map<String, dynamic> j)
      : objects = (j['objects'] as List)
            .map((o) => DetectedObject.fromJson(o))
            .toList(),
        texts = List<String>.from(j['texts']),
        guidance = j['guidance'];
}
```

---

## 6. Periyodik Otomatik Analiz (her 3 saniyede bir)

```dart
import 'dart:async';

Timer? _timer;

void startAutoAnalysis() {
  _timer = Timer.periodic(const Duration(seconds: 3), (_) async {
    await captureAndAnalyze();
  });
}

void stopAutoAnalysis() {
  _timer?.cancel();
}
```

---

## Android İzinleri

`android/app/src/main/AndroidManifest.xml` içine ekle:

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

---

## iOS İzinleri

`ios/Runner/Info.plist` içine ekle:

```xml
<key>NSCameraUsageDescription</key>
<string>SceneSpeak kameranıza erişmek istiyor</string>
<key>NSSpeechRecognitionUsageDescription</key>
<string>Sesli rehberlik için gerekli</string>
```

---

## Notlar

- `SUNUCU_IP` → bilgisayarın yerel IP adresi (ör: `192.168.1.105`)
- Telefon ve bilgisayar **aynı Wi-Fi**'da olmalı
- Bilgisayarın IP'sini öğrenmek için: `ipconfig` → "IPv4 Address"
- Backend başlatma: `uvicorn main:app --host 0.0.0.0 --port 8000`
