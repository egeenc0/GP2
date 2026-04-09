"""
tts.py
Text-to-Speech katmanı — edge-tts (Microsoft Neural) kullanır.

Desteklenen Türkçe sesler:
  tr-TR-EmelNeural   (kadın, varsayılan)
  tr-TR-AhmetNeural  (erkek)

Kullanım:
    audio_bytes = await synthesise("Merhoven saat üç yönünde kişi var.")
    # -> bytes (MP3)
"""

import asyncio
import io
import logging

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "tr-TR-EmelNeural"
FALLBACK_VOICE = "tr-TR-AhmetNeural"


async def synthesise(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """Verilen metni MP3 ses verisine dönüştürür (edge-tts).

    Hata durumunda boş bytes döner — istemci Web Speech API'ye fallback yapabilir.
    """
    try:
        import edge_tts  # geç import; kurulu değilse uygulama çökmez

        communicate = edge_tts.Communicate(text, voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        audio = buf.getvalue()
        if not audio:
            raise RuntimeError("edge-tts boş ses verisi döndürdü")
        logger.info("TTS: %d bayt üretildi (%s)", len(audio), voice)
        return audio
    except Exception as exc:
        logger.warning("TTS başarısız (%s): %s", voice, exc)
        return b""


def synthesise_sync(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """Senkron sarmalayıcı — mevcut bir event loop varsa kullanır."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # FastAPI'nin kendi loop'unda çağrılıyorsa ayrı bir coroutine olarak çalıştır
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, synthesise(text, voice))
                return future.result(timeout=15)
        return loop.run_until_complete(synthesise(text, voice))
    except Exception as exc:
        logger.warning("synthesise_sync hata: %s", exc)
        return b""
