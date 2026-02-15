# modules/tts.py
"""
Simple local TTS wrapper based on pyttsx3.
If pyttsx3 isn't available, fallback to printing.
"""

import logging
log = logging.getLogger("tts")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    pyttsx3 = None
    TTS_AVAILABLE = False

_engine = None

def tts_announce(text: str):
    if not text:
        return
    global _engine
    if not TTS_AVAILABLE:
        print("[TTS-PRINT]", text)
        return
    try:
        if _engine is None:
            _engine = pyttsx3.init()
        _engine.say(text)
        _engine.runAndWait()
    except Exception as e:
        log.warning("pyttsx3 failed: %s. Falling back to print.", e)
        print("[TTS-PRINT]", text)
