"""
camera_manager.py
Manages multiple independent camera streams.
Reuses detection helpers from detect_video.py — no detection logic duplicated.
"""
import threading
import time
from datetime import datetime
import cv2

from detect_video import (
    model,
    iou_overlap,
    _dedupe_boxes,
    _resize_for_inference,
    _PROCESS_EVERY_N,
    _COUNT_WINDOW,
)

# Registry: camera_id -> CameraStream
_cameras: dict = {}
_registry_lock = threading.Lock()

# How long (s) to wait between reconnect attempts
_RETRY_INTERVAL = 5
# Max consecutive read failures before triggering a reconnect
_MAX_READ_FAILS  = 30


def _normalise_source(source: str):
    """
    Return (parsed_source, error_string_or_None).
    Accepts:
      - integer string  "0", "1"  -> webcam index
      - rtsp://...
      - http://host:port/video     (IP Webcam full path)
      - http://host:port           -> auto-appends /video
      - http://host                -> rejects (no port)
    """
    s = str(source).strip()

    # Webcam index
    try:
        idx = int(s)
        if idx < 0:
            return None, "Webcam index must be >= 0"
        return idx, None
    except ValueError:
        pass

    # Must start with rtsp:// or http://
    if not (s.startswith("rtsp://") or s.startswith("http://")):
        return None, (
            "Source must be a webcam index (0, 1 ...), "
            "rtsp://... URL, or http://host:port/path URL"
        )

    # HTTP URL — ensure it has a port (bare http://ip fails with OpenCV)
    if s.startswith("http://"):
        from urllib.parse import urlparse
        p = urlparse(s)
        if not p.port:
            return None, (
                f"HTTP source '{s}' has no port. "
                "For IP Webcam app use: http://<ip>:8080/video"
            )
        # Auto-append /video if no path given
        if p.path in ("", "/"):
            s = s.rstrip("/") + "/video"

    return s, None


def _probe_source(source) -> bool:
    """Try to open the source and read one frame. Returns True if successful."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        return False
    ret, _ = cap.read()
    cap.release()
    return ret


def get_all_cameras():
    with _registry_lock:
        return {cid: cam.get_info() for cid, cam in _cameras.items()}


def add_camera(camera_id: str, raw_source: str):
    """Validate source, probe connectivity, then register the camera."""
    source, err = _normalise_source(raw_source)
    if err:
        return False, err

    with _registry_lock:
        if camera_id in _cameras:
            return False, "Camera ID already exists"

    # Probe before committing — gives immediate feedback on bad URLs
    if not _probe_source(source):
        hint = ""
        if isinstance(source, str) and source.startswith("http://"):
            hint = (
                " Tip: for IP Webcam app try http://<ip>:8080/video  "
                "or http://<ip>:8080/shot.jpg"
            )
        return False, f"Cannot connect to source '{source}'.{hint}"

    with _registry_lock:
        cam = CameraStream(camera_id, source)
        cam.start()
        _cameras[camera_id] = cam
    return True, "Camera added"


def remove_camera(camera_id: str):
    with _registry_lock:
        cam = _cameras.pop(camera_id, None)
    if cam:
        cam.stop()
        return True, "Camera removed"
    return False, "Camera not found"


def get_camera(camera_id: str):
    with _registry_lock:
        return _cameras.get(camera_id)


# ─────────────────────────────────────────────────────────────────────────────

class CameraStream:
    """
    One independent camera: capture thread + inference thread with auto-retry.
    """

    CONF_THRESH = 0.75

    # Possible status values surfaced to the frontend
    STATUS_CONNECTING  = "connecting"
    STATUS_LIVE        = "live"
    STATUS_RECONNECTING = "reconnecting"
    STATUS_ERROR       = "error"

    def __init__(self, camera_id: str, source):
        self.camera_id = camera_id
        self.source    = source
        self.active    = False
        self.status    = self.STATUS_CONNECTING
        self.error_msg = ""

        self._latest_frame     = None
        self._latest_annotated = None
        self._frame_lock       = threading.Lock()

        self.log: list = []
        self._log_lock = threading.Lock()

        self.stats = {"total_logs": 0, "total_violations": 0, "total_workers": 0}
        self._stats_lock = threading.Lock()

        self._count_history: list = []

    # ── public API ────────────────────────────────────────────────────── #

    def get_info(self):
        with self._stats_lock:
            s = dict(self.stats)
        return {
            "camera_id": self.camera_id,
            "source":    str(self.source),
            "active":    self.active,
            "status":    self.status,
            "error":     self.error_msg,
            "stats":     s,
        }

    def get_log(self):
        with self._log_lock:
            return list(reversed(self.log))

    def get_stats(self):
        with self._stats_lock:
            s = dict(self.stats)
        s["status"] = self.status
        s["error"]  = self.error_msg
        return s

    # ── lifecycle ─────────────────────────────────────────────────────── #

    def start(self):
        self.active = True
        self.log.clear()
        self._count_history.clear()
        with self._stats_lock:
            self.stats = {"total_logs": 0, "total_violations": 0, "total_workers": 0}
        threading.Thread(target=self._capture_loop,   daemon=True).start()
        threading.Thread(target=self._inference_loop, daemon=True).start()

    def stop(self):
        self.active = False
        self.status = self.STATUS_ERROR

    # ── capture (with auto-reconnect) ─────────────────────────────────── #

    def _capture_loop(self):
        while self.active:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.status    = self.STATUS_RECONNECTING
                self.error_msg = f"Cannot open '{self.source}'. Retrying in {_RETRY_INTERVAL}s..."
                print(f"[CameraManager:{self.camera_id}] {self.error_msg}")
                time.sleep(_RETRY_INTERVAL)
                continue

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.status    = self.STATUS_LIVE
            self.error_msg = ""
            fail_count     = 0

            while self.active:
                ret, frame = cap.read()
                if not ret:
                    fail_count += 1
                    if fail_count >= _MAX_READ_FAILS:
                        # Stream dropped — break inner loop to reconnect
                        self.status    = self.STATUS_RECONNECTING
                        self.error_msg = "Stream lost. Reconnecting..."
                        print(f"[CameraManager:{self.camera_id}] Stream lost, reconnecting...")
                        break
                    time.sleep(0.05)
                    continue

                fail_count = 0
                with self._frame_lock:
                    self._latest_frame = frame

            cap.release()
            if self.active:
                time.sleep(_RETRY_INTERVAL)

    # ── inference ─────────────────────────────────────────────────────── #

    def _inference_loop(self):
        frame_count    = 0
        last_annotated = None

        while self.active:
            with self._frame_lock:
                frame = self._latest_frame

            if frame is None:
                time.sleep(0.02)
                continue

            frame_count += 1
            if frame_count % _PROCESS_EVERY_N != 0:
                if last_annotated is not None:
                    with self._frame_lock:
                        self._latest_annotated = last_annotated
                time.sleep(0.005)
                continue

            try:
                small     = _resize_for_inference(frame)
                results   = model(small, conf=self.CONF_THRESH, verbose=False)[0]
                annotated = results.plot()
            except Exception as exc:
                print(f"[CameraManager:{self.camera_id}] Inference error: {exc}")
                time.sleep(0.1)
                continue

            with self._frame_lock:
                self._latest_annotated = annotated
            last_annotated = annotated

            names   = model.names
            boxes   = results.boxes.xyxy.cpu().numpy()
            confs   = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)

            raw_person_boxes = [boxes[i] for i, c in enumerate(classes) if names[c] == "Person"]
            raw_person_confs = [confs[i]  for i, c in enumerate(classes) if names[c] == "Person"]
            helmet_boxes     = [boxes[i]  for i, c in enumerate(classes) if names[c] == "Hardhat"]
            mask_boxes       = [boxes[i]  for i, c in enumerate(classes) if names[c] == "Mask"]

            person_boxes, _ = _dedupe_boxes(raw_person_boxes, raw_person_confs)

            self._count_history.append(len(person_boxes))
            if len(self._count_history) > _COUNT_WINDOW:
                self._count_history.pop(0)
            stable_count = round(sum(self._count_history) / len(self._count_history))

            if person_boxes:
                msg = f"[{self.camera_id}] {stable_count} Person(s) | "
                for idx, pbox in enumerate(person_boxes):
                    has_helmet = any(iou_overlap(pbox, h) for h in helmet_boxes)
                    has_mask   = any(iou_overlap(pbox, m) for m in mask_boxes)
                    msg += f"P{idx+1}: "
                    msg += ("OK Helmet" if has_helmet else "No Helmet") + ", "
                    msg += ("OK Mask"   if has_mask   else "No Mask")
                    if idx < len(person_boxes) - 1:
                        msg += " | "
            else:
                msg = f"[{self.camera_id}] No person detected"

            category = "violation" if ("No Helmet" in msg or "No Mask" in msg) else "normal"
            entry = {
                "camera_id": self.camera_id,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message":   msg,
                "category":  category,
            }

            with self._log_lock:
                self.log.append(entry)
                if len(self.log) > 30:
                    self.log.pop(0)

            with self._stats_lock:
                self.stats["total_logs"] += 1
                if category == "violation":
                    self.stats["total_violations"] += 1
                if stable_count > self.stats["total_workers"]:
                    self.stats["total_workers"] = stable_count

    # ── MJPEG stream ──────────────────────────────────────────────────── #

    def generate_feed(self):
        """MJPEG generator for Flask Response. Streams error frame when offline."""
        timeout = time.time() + 8
        while self._latest_annotated is None and self.active and time.time() < timeout:
            time.sleep(0.05)

        while self.active:
            with self._frame_lock:
                frame = self._latest_annotated

            if frame is None:
                # Emit a black placeholder frame so the <img> tag doesn't break
                import numpy as np
                placeholder = np.zeros((240, 426, 3), dtype="uint8")
                cv2.putText(placeholder, self.status.upper(), (80, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
                ret, buf = cv2.imencode(".jpg", placeholder)
            else:
                ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

            if ret:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
            time.sleep(0.1)
