# =============================================================================
# [NEW MODULE] alert_manager.py
# Real-Time Alert System for PPE Violations
# Handles: Email (SMTP) + Telegram Bot alerts
# Design:  Non-blocking (threaded), cooldown-protected, config-driven
# =============================================================================

import os
import time
import threading
import smtplib
import logging
import csv
import requests
import cv2

from email.mime.multipart import MIMEMultipart
from email.mime.text     import MIMEText
from email.mime.image    import MIMEImage
from datetime            import datetime
from dotenv              import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env (must sit next to this file)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Feature flags  — set to "true" / "false" in .env
# ---------------------------------------------------------------------------
ALERT_EMAIL_ENABLED    = os.getenv("ALERT_EMAIL_ENABLED",    "false").lower() == "true"
ALERT_TELEGRAM_ENABLED = os.getenv("ALERT_TELEGRAM_ENABLED", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Email config
# ---------------------------------------------------------------------------
SMTP_HOST     = os.getenv("SMTP_HOST",     "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER",     "")          # sender address
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")          # app password
ALERT_TO      = os.getenv("ALERT_TO",      "")          # recipient address

# ---------------------------------------------------------------------------
# Telegram config
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")

# ---------------------------------------------------------------------------
# Alert behaviour
# ---------------------------------------------------------------------------
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "10"))
ALERT_LOCATION         = os.getenv("ALERT_LOCATION", "Zone A")          # static label
ALERT_LOG_FILE         = os.getenv("ALERT_LOG_FILE", "violation_logs/alert_log.csv")

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AlertManager] %(levelname)s: %(message)s",
)
_log = logging.getLogger("alert_manager")

# ADDED: one-shot flag — set by _send_email on success, cleared by /alert_status
_email_sent_event = threading.Event()

def consume_email_sent_flag() -> bool:
    """Returns True once per successful email send, then resets. Thread-safe."""
    if _email_sent_event.is_set():
        _email_sent_event.clear()
        return True
    return False

# ---------------------------------------------------------------------------
# Cooldown tracker  { violation_key -> last_alert_epoch }
# Protected by a lock so the inference thread and any future callers are safe
# ---------------------------------------------------------------------------
_cooldown_map:  dict  = {}
_cooldown_lock: threading.Lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_on_cooldown(key: str) -> bool:
    """Return True if this violation type was alerted within the cooldown window."""
    with _cooldown_lock:
        last = _cooldown_map.get(key, 0)
        return (time.time() - last) < ALERT_COOLDOWN_SECONDS


def _mark_alerted(key: str) -> None:
    """Record the current time as the last-alert time for this key."""
    with _cooldown_lock:
        _cooldown_map[key] = time.time()


def _encode_frame_to_bytes(frame) -> bytes | None:
    """JPEG-encode an OpenCV frame; return raw bytes or None on failure."""
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


def _log_alert(violation_type: str, timestamp: str, location: str) -> None:
    """Append one row to the CSV alert log (creates file + header if needed)."""
    os.makedirs(os.path.dirname(ALERT_LOG_FILE), exist_ok=True)
    file_exists = os.path.isfile(ALERT_LOG_FILE)
    try:
        with open(ALERT_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "ViolationType", "Location", "Channel"])
            writer.writerow([timestamp, violation_type, location, "email+telegram"])
    except OSError as exc:
        _log.warning("Could not write alert log: %s", exc)


# ---------------------------------------------------------------------------
# Email alert
# ---------------------------------------------------------------------------

def _send_email(violation_type: str, timestamp: str, location: str, frame_bytes: bytes | None) -> None:
    """
    Build and send an HTML email with an optional inline snapshot.
    Runs inside a daemon thread — never called directly by the caller.
    """
    if not all([SMTP_USER, SMTP_PASSWORD, ALERT_TO]):
        _log.warning("Email credentials incomplete — skipping email alert.")
        return

    subject = f"⚠️ PPE Violation Detected — {violation_type}"

    html_body = f"""
    <html><body>
    <h2 style="color:#c0392b;">⚠️ PPE Safety Violation Alert</h2>
    <table cellpadding="8" style="border-collapse:collapse;">
      <tr><td><b>Timestamp</b></td><td>{timestamp}</td></tr>
      <tr><td><b>Violation</b></td><td>{violation_type}</td></tr>
      <tr><td><b>Location</b></td><td>{location}</td></tr>
    </table>
    {"<br><img src='cid:snapshot' style='max-width:640px;border:2px solid #c0392b;border-radius:6px;'/>" if frame_bytes else ""}
    <p style="color:#7f8c8d;font-size:12px;">Sent by AI CCTV PPE Detection System</p>
    </body></html>
    """

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"]    = SMTP_USER
    msg["To"]      = ALERT_TO

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(html_body, "html"))
    msg.attach(alt)

    if frame_bytes:
        img_part = MIMEImage(frame_bytes, name="snapshot.jpg")
        img_part.add_header("Content-ID", "<snapshot>")
        img_part.add_header("Content-Disposition", "inline", filename="snapshot.jpg")
        msg.attach(img_part)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, ALERT_TO, msg.as_string())
        _log.info("Email alert sent → %s", ALERT_TO)
        _email_sent_event.set()  # ADDED: signal frontend that email was sent
    except Exception as exc:                          # broad catch: network, auth, etc.
        _log.error("Email send failed: %s", exc)


# ---------------------------------------------------------------------------
# Telegram alert
# ---------------------------------------------------------------------------

def _send_telegram(violation_type: str, timestamp: str, location: str, frame_bytes: bytes | None) -> None:
    """
    Send a Telegram message (+ optional photo) via the Bot API.
    Runs inside a daemon thread — never called directly by the caller.
    """
    if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        _log.warning("Telegram credentials incomplete — skipping Telegram alert.")
        return

    text = (
        f"⚠️ *PPE Violation Detected*\n"
        f"🕐 *Time:* {timestamp}\n"
        f"🚨 *Violation:* {violation_type}\n"
        f"📍 *Location:* {location}"
    )

    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

    try:
        if frame_bytes:
            # Send photo with caption
            resp = requests.post(
                f"{base_url}/sendPhoto",
                data={
                    "chat_id":    TELEGRAM_CHAT_ID,
                    "caption":    text,
                    "parse_mode": "Markdown",
                },
                files={"photo": ("snapshot.jpg", frame_bytes, "image/jpeg")},
                timeout=10,
            )
        else:
            # Text-only message
            resp = requests.post(
                f"{base_url}/sendMessage",
                json={
                    "chat_id":    TELEGRAM_CHAT_ID,
                    "text":       text,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )

        if resp.ok:
            _log.info("Telegram alert sent (chat_id=%s)", TELEGRAM_CHAT_ID)
        else:
            _log.error("Telegram API error %s: %s", resp.status_code, resp.text)

    except requests.RequestException as exc:
        _log.error("Telegram send failed: %s", exc)


# ---------------------------------------------------------------------------
# Public API — the ONLY function imported by detect_video.py
# ---------------------------------------------------------------------------

def trigger_alert(violation_type: str, frame=None, metadata: dict | None = None) -> None:
    """
    Entry point called by detect_video._inference_thread() when a violation
    is confirmed.

    Parameters
    ----------
    violation_type : str
        Human-readable description, e.g. "No Helmet (P1), No Mask (P2)".
    frame : numpy.ndarray | None
        The annotated OpenCV frame at the moment of violation (for snapshot).
    metadata : dict | None
        Optional extra context, e.g. {"location": "Zone B"}.

    Behaviour
    ---------
    - Checks cooldown: if the same violation key was alerted within
      ALERT_COOLDOWN_SECONDS, the call is silently dropped.
    - Dispatches email and/or Telegram in separate daemon threads so the
      video pipeline is never blocked.
    - Logs every dispatched alert to the CSV alert log.
    """
    if not (ALERT_EMAIL_ENABLED or ALERT_TELEGRAM_ENABLED):
        return                                        # all channels disabled — fast exit

    # Use violation_type as the cooldown key (coarse-grained dedup)
    cooldown_key = violation_type
    if _is_on_cooldown(cooldown_key):
        return                                        # within cooldown window — skip

    _mark_alerted(cooldown_key)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    location  = (metadata or {}).get("location", ALERT_LOCATION)

    # Encode frame once; share bytes across both channels
    frame_bytes = _encode_frame_to_bytes(frame)

    _log.info("Triggering alert | violation=%s | location=%s", violation_type, location)
    _log_alert(violation_type, timestamp, location)

    # --- Dispatch channels in background daemon threads ---
    if ALERT_EMAIL_ENABLED:
        threading.Thread(
            target=_send_email,
            args=(violation_type, timestamp, location, frame_bytes),
            daemon=True,
            name="alert-email",
        ).start()

    if ALERT_TELEGRAM_ENABLED:
        threading.Thread(
            target=_send_telegram,
            args=(violation_type, timestamp, location, frame_bytes),
            daemon=True,
            name="alert-telegram",
        ).start()
