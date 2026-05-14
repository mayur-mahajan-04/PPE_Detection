from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import threading
import time

# [NEW] Real-Time Alert System — import the public trigger function only
from alert_manager import trigger_alert
from voice_alert import speak_violation

model = YOLO("models/best.pt")

# Global state
LIVE_FEED_ACTIVE = False
camera_released = True
DETECTION_LOG = []
detection_thread = None
cap = None

# Thread-safe cumulative stats
_stats_lock = threading.Lock()
STATS = {"total_logs": 0, "total_violations": 0, "total_workers": 0, "fps": 0.0}

# --- Shared frame buffer ---
_latest_frame     = None
_latest_annotated = None
_frame_lock       = threading.Lock()

# --- Tunable constants ---
_PROCESS_EVERY_N  = 2       # infer every Nth frame
_INFER_WIDTH      = 640     # resize width before inference
_DEDUP_IOU_THRESH = 0.45    # NMS IoU threshold for persons
_COUNT_WINDOW     = 5       # rolling window for person-count smoothing
_CONF_THRESH      = 0.50    # YOLO confidence threshold (lowered for better recall)
_VIOLATION_CONFIRM = 3      # frames a violation must persist before logging

_count_history    = []
_violation_buffer = []      # last N violation flags for smoothing


# ── Preprocessing ────────────────────────────────────────────────────────────
def _preprocess(frame):
    """
    Resize to _INFER_WIDTH, apply CLAHE for contrast enhancement,
    then mild Gaussian blur to reduce sensor noise.
    Returns BGR frame ready for YOLO.
    """
    h, w = frame.shape[:2]
    if w != _INFER_WIDTH:
        scale = _INFER_WIDTH / w
        frame = cv2.resize(frame, (_INFER_WIDTH, int(h * scale)),
                           interpolation=cv2.INTER_LINEAR)
    # CLAHE on L-channel of LAB for contrast boost without colour shift
    lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    # Mild blur to suppress noise without losing edges
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    return frame


# kept as alias so camera_manager.py import still works
def _resize_for_inference(frame):
    return _preprocess(frame)


# ── IoU / NMS helpers ────────────────────────────────────────────────────────
def iou_overlap(boxA, boxB, threshold: float = 0.2) -> bool:
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter  = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return False
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return (inter / union) > threshold


def _dedupe_boxes(boxes, confs, iou_thresh=_DEDUP_IOU_THRESH):
    """Greedy NMS for a single class."""
    if len(boxes) <= 1:
        return boxes, confs
    order = sorted(range(len(boxes)), key=lambda i: confs[i], reverse=True)
    kept  = []
    while order:
        best = order.pop(0)
        kept.append(best)
        order = [i for i in order
                 if not iou_overlap(boxes[best], boxes[i], threshold=iou_thresh)]
    return [boxes[i] for i in kept], [confs[i] for i in kept]


def _capture_thread():
    """
    Optimization: dedicated capture thread writes raw frames to _latest_frame.
    Decouples capture latency from processing latency.
    """
    global _latest_frame, cap, LIVE_FEED_ACTIVE, camera_released

    # Optimization: open capture once here instead of in generate_live_feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    # Optimization: minimize internal OpenCV buffer to reduce stale-frame lag
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    camera_released = False

    while LIVE_FEED_ACTIVE:
        ret, frame = cap.read()
        if not ret:
            break
        # Optimization: overwrite without copying — consumers always get the latest frame
        with _frame_lock:
            _latest_frame = frame

    cap.release()
    camera_released = True


def _inference_thread():
    """
    Optimization: runs YOLO inference on resized frames at a controlled rate,
    writes annotated result to _latest_annotated for the streaming generator.
    Also handles detection logging (replaces detection_worker reads from cap).
    """
    global _latest_annotated, DETECTION_LOG, LIVE_FEED_ACTIVE

    CONF_THRESH = 0.75
    frame_count = 0
    last_annotated = None

    while LIVE_FEED_ACTIVE:
        with _frame_lock:
            frame = _latest_frame

        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1

        # Optimization: skip frames — only infer every Nth frame
        if frame_count % _PROCESS_EVERY_N != 0:
            # Reuse last annotated frame so stream stays smooth
            if last_annotated is not None:
                with _frame_lock:
                    _latest_annotated = last_annotated
            time.sleep(0.005)
            continue

        # Optimization: infer on a smaller resolution copy
        small = _resize_for_inference(frame)
        results = model(small, conf=CONF_THRESH, verbose=False)[0]
        annotated = results.plot()

        with _frame_lock:
            _latest_annotated = annotated
        last_annotated = annotated

        # --- Detection logging (runs at inference rate, not every frame) ---
        names   = model.names
        boxes   = results.boxes.xyxy.cpu().numpy()
        confs   = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        # Raw per-class extraction
        raw_person_boxes = [boxes[i] for i, c in enumerate(classes) if names[c] == "Person"]
        raw_person_confs = [confs[i]  for i, c in enumerate(classes) if names[c] == "Person"]
        helmet_boxes     = [boxes[i]  for i, c in enumerate(classes) if names[c] == "Hardhat"]
        mask_boxes       = [boxes[i]  for i, c in enumerate(classes) if names[c] == "Mask"]

        # Secondary NMS: remove duplicate boxes for the same physical person
        person_boxes, _ = _dedupe_boxes(raw_person_boxes, raw_person_confs)

        # Temporal smoothing: 3-frame rolling average prevents single-frame count spikes
        _count_history.append(len(person_boxes))
        if len(_count_history) > _COUNT_WINDOW:
            _count_history.pop(0)
        stable_count = round(sum(_count_history) / len(_count_history))

        message = ""
        if person_boxes:
            message += f"👷 {stable_count} Person(s) detected | "
            for idx, pbox in enumerate(person_boxes):
                has_helmet = any(iou_overlap(pbox, hbox) for hbox in helmet_boxes)
                has_mask   = any(iou_overlap(pbox, mbox) for mbox in mask_boxes)
                message += f"👤P{idx+1}: "
                message += "✅ Helmet, " if has_helmet else "❌ No Helmet, "
                message += "✅ Mask"     if has_mask   else "❌ No Mask"
                if idx < len(person_boxes) - 1:
                    message += " | "
        else:
            message += "❌ No person detected"

        log_entry = {
            "id": str(datetime.now().timestamp()),
            "timestamp": datetime.now().strftime('%H:%M:%S'),
            "message": message,
            "category": "violation" if "❌" in message else "normal",
        }
        DETECTION_LOG.append(log_entry)
        if len(DETECTION_LOG) > 20:
            DETECTION_LOG.pop(0)

        # Update cumulative stats atomically
        with _stats_lock:
            STATS["total_logs"] += 1
            if log_entry["category"] == "violation":
                STATS["total_violations"] += 1
            # Track peak worker count seen this session, not a running sum
            if stable_count > STATS["total_workers"]:
                STATS["total_workers"] = stable_count

        # [NEW] Fire alert when any PPE violation is present in this frame.
        # trigger_alert() is non-blocking (spawns daemon threads internally)
        # and cooldown-protected, so calling it every inference cycle is safe.
        if log_entry["category"] == "violation" and person_boxes:
            # Build a compact violation summary, e.g. "No Helmet (P1) | No Mask (P2)"
            violation_parts = []
            for idx, pbox in enumerate(person_boxes):
                has_helmet = any(iou_overlap(pbox, hbox) for hbox in helmet_boxes)
                has_mask   = any(iou_overlap(pbox, mbox) for mbox in mask_boxes)
                issues = []
                if not has_helmet:
                    issues.append("No Helmet")
                if not has_mask:
                    issues.append("No Mask")
                if issues:
                    violation_parts.append(f"{', '.join(issues)} (P{idx+1})")
            if violation_parts:
                vtype = " | ".join(violation_parts)
                trigger_alert(
                    violation_type=vtype,
                    frame=annotated,
                    metadata={"person_count": stable_count},
                )
                speak_violation(vtype)   # non-blocking voice alert

        # Optimization: removed print() from hot loop to avoid I/O stalls


def generate_live_feed():
    """Generates MJPEG frames for streaming — reads from shared annotated buffer."""
    global LIVE_FEED_ACTIVE

    # Wait briefly for the capture thread to produce the first frame
    timeout = time.time() + 5
    while _latest_annotated is None and time.time() < timeout:
        time.sleep(0.02)

    while LIVE_FEED_ACTIVE:
        with _frame_lock:
            annotated = _latest_annotated

        if annotated is None:
            time.sleep(0.01)
            continue

        # Optimization: encode with slightly reduced JPEG quality for faster transfer
        ret, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        # Optimization: cap stream rate to ~30 FPS max to avoid busy-looping
        time.sleep(0.033)


def detection_worker():
    """Kept for API compatibility — logic merged into _inference_thread."""
    pass


def start_live():
    global LIVE_FEED_ACTIVE, detection_thread, DETECTION_LOG, _latest_frame, _latest_annotated
    if not LIVE_FEED_ACTIVE:
        LIVE_FEED_ACTIVE = True
        DETECTION_LOG = []
        _latest_frame = None
        _latest_annotated = None
        # Reset stats on every camera restart
        with _stats_lock:
            STATS["total_logs"] = 0
            STATS["total_violations"] = 0
            STATS["total_workers"] = 0

        # Optimization: two focused threads — one for capture, one for inference+logging
        threading.Thread(target=_capture_thread,   daemon=True).start()
        threading.Thread(target=_inference_thread, daemon=True).start()

        print("🚀 Live detection started")


def stop_live():
    global LIVE_FEED_ACTIVE, detection_thread
    LIVE_FEED_ACTIVE = False

    if detection_thread is not None:
        detection_thread.join(timeout=2)

    print("🛑 Live detection stopped")
