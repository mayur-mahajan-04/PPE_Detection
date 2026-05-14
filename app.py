import os
import uuid
import cv2
from ultralytics import YOLO
from flask import Flask, send_file, render_template, Response, request, jsonify
from detect_video import (
    generate_live_feed,
    start_live,
    stop_live,
    DETECTION_LOG,
    STATS,
    _stats_lock
)
from alert_manager import consume_email_sent_flag  # ADDED
from waitress import serve  # ✅ Import waitress for production server

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("landing.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/start", methods=["POST"])
def start():
    start_live()
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop():
    stop_live()
    return jsonify({"status": "stopped"})

@app.route("/video_feed")
def video_feed():
    return Response(generate_live_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_feed/<camera_id>")
def video_feed_by_id(camera_id):
    """Unified feed endpoint: 'cam0' = built-in webcam, others = camera_manager."""
    if camera_id == "cam0":
        return Response(generate_live_feed(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    from camera_manager import get_camera
    cam = get_camera(camera_id)
    if not cam:
        return "Camera not found", 404
    return Response(cam.generate_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/detection_log")
def detection_log():
    from detect_video import DETECTION_LOG
    # Return the logs in reverse order (newest first)
    return jsonify({"logs": list(reversed(DETECTION_LOG))})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_file():
    from PIL import Image

    file = request.files["file"]
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    model = YOLO("models/best.pt")

    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        # Handle image
        img = cv2.imread(path)
        results = model(img)[0]
        result_img = results.plot()

        output_path = os.path.join(UPLOAD_FOLDER, f"result_{filename}")
        cv2.imwrite(output_path, result_img)

        return send_file(output_path, mimetype="image/jpeg")

    elif filename.lower().endswith((".mp4", ".avi", ".mov")):
        # Handle video
        cap = cv2.VideoCapture(path)
        output_path = os.path.join(UPLOAD_FOLDER, f"result_{filename}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)[0]
            annotated = results.plot()
            out.write(annotated)

        cap.release()
        out.release()

        return send_file(output_path, mimetype="video/mp4")

    else:
        return "Unsupported file type", 400

# ADDED: lightweight endpoint polled by frontend to detect email send events
@app.route("/alert_status")
def alert_status():
    sent = consume_email_sent_flag()  # True once per real send, then resets
    return jsonify({"alert": "email_sent" if sent else "none"})

@app.route("/stats")
def stats():
    from detect_video import STATS, _stats_lock
    with _stats_lock:
        snapshot = dict(STATS)
    return jsonify(snapshot)

@app.route("/export_logs")
def export_logs():
    from detect_video import DETECTION_LOG
    import csv
    from io import StringIO
    import html
    
    # Create CSV data in memory with UTF-8 encoding
    csv_data = StringIO()
    writer = csv.writer(csv_data)
    
    # Write UTF-8 BOM for Excel compatibility
    csv_data.write('\ufeff')
    
    # Write header
    writer.writerow(["Timestamp", "Category", "Status", "Details"])
    
    # Write log data
    for log in reversed(DETECTION_LOG):
        # Clean and format the message
        clean_message = log['message'].replace('👷', 'Workers:')
        clean_message = log['message'].replace('👤', 'Person')
        clean_message = log['message'].replace('✅', 'Yes')
        clean_message = log['message'].replace('❌', 'No')
        
        # Split into status and details
        if "|" in clean_message:
            status, details = clean_message.split("|", 1)
        else:
            status = clean_message
            details = ""
            
        writer.writerow([
            log['timestamp'],
            log['category'].capitalize(),
            status.strip(),
            details.strip()
        ])
    
    # Create response with CSV data
    response = Response(
        csv_data.getvalue(),
        mimetype="text/csv; charset=utf-8-sig",
        headers={
            "Content-disposition": "attachment; filename=ppe_detection_logs.csv"
        }
    )
    
    return response

# ── Voice Alert routes ──────────────────────────────────────────────────────
from voice_alert import set_enabled as voice_set_enabled, get_status as voice_get_status

@app.route("/voice/toggle", methods=["POST"])
def voice_toggle():
    data    = request.get_json(force=True) or {}
    enabled = bool(data.get("enabled", True))
    voice_set_enabled(enabled)
    return jsonify(voice_get_status())

@app.route("/voice/status")
def voice_status():
    return jsonify(voice_get_status())
# ─────────────────────────────────────────────────────────────────────────────

# ── Multi-Camera routes ──────────────────────────────────────────────────────
from camera_manager import add_camera, remove_camera, get_all_cameras, get_camera

@app.route("/cameras")
def cameras_list():
    return jsonify(get_all_cameras())

@app.route("/camera/add", methods=["POST"])
def camera_add():
    data      = request.get_json(force=True)
    camera_id = data.get("camera_id", "").strip()
    source    = data.get("source", "").strip()
    if not camera_id or source == "":
        return jsonify({"ok": False, "message": "camera_id and source are required"}), 400
    ok, msg = add_camera(camera_id, source)
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 400)

@app.route("/camera/remove/<camera_id>", methods=["POST"])
def camera_remove(camera_id):
    ok, msg = remove_camera(camera_id)
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 404)

@app.route("/camera/feed/<camera_id>")
def camera_feed(camera_id):
    cam = get_camera(camera_id)
    if not cam:
        return "Camera not found", 404
    return Response(cam.generate_feed(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/camera/snapshot/<camera_id>")
def camera_snapshot(camera_id):
    """Returns a single JPEG frame for thumbnail use (no persistent connection)."""
    if camera_id == "cam0":
        from detect_video import _latest_annotated, _frame_lock
        import numpy as np
        with _frame_lock:
            frame = _latest_annotated
        if frame is None:
            frame = np.zeros((72, 128, 3), dtype="uint8")
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not ret:
            return "", 204
        return Response(buf.tobytes(), mimetype="image/jpeg")
    cam = get_camera(camera_id)
    if not cam:
        return "", 404
    import threading
    with cam._frame_lock:
        frame = cam._latest_annotated
    if frame is None:
        import numpy as np
        frame = np.zeros((72, 128, 3), dtype="uint8")
    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if not ret:
        return "", 204
    return Response(buf.tobytes(), mimetype="image/jpeg")

@app.route("/camera/log/<camera_id>")
def camera_log(camera_id):
    cam = get_camera(camera_id)
    if not cam:
        return jsonify({"logs": []})
    return jsonify({"logs": cam.get_log()})

@app.route("/camera/stats/<camera_id>")
def camera_stats(camera_id):
    cam = get_camera(camera_id)
    if not cam:
        return jsonify({"total_logs": 0, "total_violations": 0, "total_workers": 0})
    return jsonify(cam.get_stats())
# ─────────────────────────────────────────────────────────────────────────────

# ── Analytics API routes — powered by live DETECTION_LOG + STATS ────────────
@app.route("/api/stats/summary")
def api_stats_summary():
    from detect_video import DETECTION_LOG, STATS, _stats_lock
    with _stats_lock:
        snap = dict(STATS)
    logs = list(DETECTION_LOG)          # snapshot to avoid race
    no_helmet = sum(1 for e in logs if "No Helmet" in e["message"])
    no_mask   = sum(1 for e in logs if "No Mask"   in e["message"])
    both      = sum(1 for e in logs if "No Helmet" in e["message"] and "No Mask" in e["message"])
    return jsonify({
        "total":      snap["total_logs"],
        "violations": snap["total_violations"],
        "workers":    snap["total_workers"],
        "no_helmet":  no_helmet,
        "no_mask":    no_mask,
        "both":       both,
    })

@app.route("/api/stats/hourly")
def api_stats_hourly():
    """Returns per-minute violation counts from the last 20 log entries."""
    from detect_video import DETECTION_LOG
    logs = list(DETECTION_LOG)
    if not logs:
        return jsonify({"labels": [], "values": []})
    # Group by HH:MM timestamp
    counts = {}
    for e in logs:
        if e["category"] == "violation":
            minute = e["timestamp"][:5]   # "HH:MM"
            counts[minute] = counts.get(minute, 0) + 1
    if not counts:
        return jsonify({"labels": [], "values": []})
    labels = sorted(counts.keys())
    return jsonify({"labels": labels, "values": [counts[l] for l in labels]})

@app.route("/api/stats/by_type")
def api_stats_by_type():
    from detect_video import DETECTION_LOG
    logs = list(DETECTION_LOG)
    if not logs:
        return jsonify({"labels": ["No Data"], "values": [1]})
    helmet_only = sum(1 for e in logs if "No Helmet" in e["message"] and "No Mask" not in e["message"])
    mask_only   = sum(1 for e in logs if "No Mask"   in e["message"] and "No Helmet" not in e["message"])
    both        = sum(1 for e in logs if "No Helmet" in e["message"] and "No Mask"   in e["message"])
    compliant   = sum(1 for e in logs if e["category"] == "normal" and "Person" in e["message"])
    counts = {}
    if helmet_only: counts["No Helmet"]      = helmet_only
    if mask_only:   counts["No Mask"]        = mask_only
    if both:        counts["No Helmet+Mask"] = both
    if compliant:   counts["Compliant"]      = compliant
    if not counts:  counts["No Data"]        = 1
    return jsonify({"labels": list(counts.keys()), "values": list(counts.values())})

@app.route("/api/stats/timeline")
def api_stats_timeline():
    """
    Returns a rolling 60-second timeline of violation counts.
    Each bucket = 1 second. Always returns exactly 60 buckets so the
    line chart scrolls smoothly without resizing.
    """
    from detect_video import DETECTION_LOG, STATS, _stats_lock
    from datetime import datetime, timedelta
    logs = list(DETECTION_LOG)
    now  = datetime.now()

    # Build 60 one-second buckets labelled HH:MM:SS
    buckets = {}
    for i in range(59, -1, -1):
        t = now - timedelta(seconds=i)
        buckets[t.strftime("%H:%M:%S")] = 0

    for e in logs:
        if e["category"] == "violation":
            ts = e["timestamp"]   # already "HH:MM:SS"
            if ts in buckets:
                buckets[ts] += 1

    with _stats_lock:
        fps = STATS.get("fps", 0.0)

    labels = list(buckets.keys())
    values = list(buckets.values())
    return jsonify({"labels": labels, "values": values, "fps": fps})
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Starting app with Waitress on http://localhost:8000")
    serve(app, host='0.0.0.0', port=8000)
