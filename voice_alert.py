"""
voice_alert.py
==============
Offline text-to-speech voice alerts for PPE violations.

Design
------
- Single background daemon thread owns the pyttsx3 engine (engines are NOT
  thread-safe; keeping one thread avoids all race conditions).
- Callers push messages onto a queue — the detection pipeline is never blocked.
- Per-violation-type cooldown prevents repeated announcements.
- Global enable/disable flag toggled at runtime via /voice/toggle API.

Dependencies
------------
    pip install pyttsx3

Configuration (via .env or environment variables)
-------------------------------------------------
    VOICE_ENABLED          true | false   (default: true)
    VOICE_COOLDOWN_SECONDS int            (default: 8)
    VOICE_RATE             int wpm        (default: 160)
    VOICE_VOLUME           0.0–1.0        (default: 0.9)
"""

import queue
import threading
import time
import logging
import os

from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
VOICE_ENABLED          = os.getenv("VOICE_ENABLED",          "true").lower() == "true"
VOICE_COOLDOWN_SECONDS = int(os.getenv("VOICE_COOLDOWN_SECONDS", "8"))
VOICE_RATE             = int(os.getenv("VOICE_RATE",             "160"))
VOICE_VOLUME           = float(os.getenv("VOICE_VOLUME",         "0.9"))

# ── Violation → human-readable phrase map ─────────────────────────────────────
_PHRASES = {
    "no helmet":       "Warning! Helmet not detected.",
    "no mask":         "Warning! Mask not detected.",
    "no helmet+mask":  "Warning! Helmet and mask not detected.",
    "no helmet, no mask": "Warning! Helmet and mask not detected.",
}

_DEFAULT_PHRASE = "Warning! PPE violation detected."

# ── Module state ──────────────────────────────────────────────────────────────
_enabled       = VOICE_ENABLED
_enabled_lock  = threading.Lock()

_msg_queue: queue.Queue = queue.Queue(maxsize=10)   # bounded — drops if full

# cooldown: violation_key (lowercase) -> last_spoken epoch
_cooldown_map:  dict           = {}
_cooldown_lock: threading.Lock = threading.Lock()

_log = logging.getLogger("voice_alert")

# ── TTS worker thread ─────────────────────────────────────────────────────────

def _tts_worker():
    """
    Runs forever in a daemon thread.
    Owns the pyttsx3 engine exclusively — no other thread touches it.
    """
    try:
        import pyttsx3
    except ImportError:
        _log.error("pyttsx3 not installed. Run: pip install pyttsx3")
        return

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate",   VOICE_RATE)
        engine.setProperty("volume", VOICE_VOLUME)
    except Exception as exc:
        _log.error("pyttsx3 init failed: %s", exc)
        return

    _log.info("Voice alert engine ready (rate=%d, volume=%.1f)", VOICE_RATE, VOICE_VOLUME)

    while True:
        try:
            phrase = _msg_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            engine.say(phrase)
            engine.runAndWait()
        except Exception as exc:
            _log.warning("TTS speak error: %s", exc)
        finally:
            _msg_queue.task_done()


# Start the worker once at import time
_worker_thread = threading.Thread(target=_tts_worker, daemon=True, name="voice-tts")
_worker_thread.start()


# ── Cooldown helpers ──────────────────────────────────────────────────────────

def _on_cooldown(key: str) -> bool:
    with _cooldown_lock:
        return (time.time() - _cooldown_map.get(key, 0)) < VOICE_COOLDOWN_SECONDS


def _mark(key: str) -> None:
    with _cooldown_lock:
        _cooldown_map[key] = time.time()


# ── Phrase builder ────────────────────────────────────────────────────────────

def _build_phrase(violation_type: str) -> str:
    """
    Map a violation_type string (e.g. 'No Helmet (P1) | No Mask (P2)')
    to a clear spoken phrase.
    """
    vt = violation_type.lower()

    has_helmet = "no helmet" in vt
    has_mask   = "no mask"   in vt

    if has_helmet and has_mask:
        return _PHRASES["no helmet+mask"]
    if has_helmet:
        return _PHRASES["no helmet"]
    if has_mask:
        return _PHRASES["no mask"]

    # Fallback: look up exact key
    for key, phrase in _PHRASES.items():
        if key in vt:
            return phrase

    return _DEFAULT_PHRASE


# ── Public API ────────────────────────────────────────────────────────────────

def speak_violation(violation_type: str) -> None:
    """
    Called by detect_video / camera_manager when a confirmed violation occurs.
    Non-blocking: pushes to queue and returns immediately.
    Silently dropped if:
      - voice alerts are disabled
      - same violation type is within cooldown window
      - queue is full (system under load)
    """
    with _enabled_lock:
        if not _enabled:
            return

    key = violation_type.lower().strip()
    if _on_cooldown(key):
        return

    phrase = _build_phrase(violation_type)
    _mark(key)

    try:
        _msg_queue.put_nowait(phrase)
        _log.debug("Queued voice alert: %s", phrase)
    except queue.Full:
        _log.debug("Voice queue full — alert dropped")


def set_enabled(state: bool) -> None:
    """Enable or disable voice alerts at runtime."""
    global _enabled
    with _enabled_lock:
        _enabled = state
    _log.info("Voice alerts %s", "enabled" if state else "disabled")


def get_status() -> dict:
    """Return current voice alert status for the API."""
    with _enabled_lock:
        enabled = _enabled
    return {
        "enabled":          enabled,
        "cooldown_seconds": VOICE_COOLDOWN_SECONDS,
        "rate":             VOICE_RATE,
        "volume":           VOICE_VOLUME,
        "queue_size":       _msg_queue.qsize(),
    }
