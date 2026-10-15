import os
import time
import traceback
from datetime import datetime, timedelta
from threading import Lock
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

import config
from models.efficientdet import EfficientDetDetector
from models.ensemble import EnsembleModel
from models.faster_rcnn import FasterRCNNDetector

app = Flask(__name__)

UPLOAD_DIR = os.path.join(app.static_folder, "uploads")
PROCESSED_DIR = os.path.join(app.static_folder, "processed")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

print("[SYSTEM] Initializing Neural Networks...")
faster_rcnn = FasterRCNNDetector()
efficient_det = EfficientDetDetector()
ensemble_fusion = EnsembleModel()
print("[SYSTEM] All Systems Online. Ready for Surveillance.")

threat_lock = Lock()
last_webcam_threat = False
last_webcam_confidence = 0.0
last_webcam_threat_type = None
log_lock = Lock()
detection_logs = []
scan_timestamps = []
app_started_at = datetime.now()
MAX_LOG_ENTRIES = 5000
LIVE_LOG_INTERVAL_SECONDS = 3
last_live_log_time = None
VIDEO_FRAME_SKIP = 5
INFERENCE_WIDTH = 640
INFERENCE_HEIGHT = 480
LIVE_INFERENCE_SIZE = 416
LIVE_DETECTION_INTERVAL = 12
LIVE_SCENE_CHANGE_THRESHOLD = 18.0

API_JSON_ROUTES = {"/upload", "/detect", "/threat_status"}


@app.errorhandler(Exception)
def handle_unexpected_exception(error):
    if request.path in API_JSON_ROUTES:
        print(f"[ERROR] API failure on {request.path}: {error}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(error)}), 500

    print(f"[ERROR] Unhandled failure on {request.path}: {error}")
    traceback.print_exc()
    return "Internal Server Error", 500


def _is_person_label(label):
    return int(label) == int(config.PERSON_CLASS_ID)


def _is_weapon_label(label):
    return int(label) in config.WEAPON_CLASSES


THREAT_DECISION_THRESHOLD = float(config.CONFIDENCE_THRESHOLD)
PERSON_CONFIDENCE_THRESHOLD = 0.75
# Cross-model deduplication IoU: used to suppress duplicate boxes from different detectors.
CROSS_MODEL_NMS_IOU_THRESHOLD = 0.30
PERSON_NMS_IOU_THRESHOLD = CROSS_MODEL_NMS_IOU_THRESHOLD
WEAPON_NMS_IOU_THRESHOLD = CROSS_MODEL_NMS_IOU_THRESHOLD


def _bbox_center(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def is_weapon_center_inside_person(weapon_box, person_box):
    wx, wy = _bbox_center(weapon_box)
    px1, py1, px2, py2 = [float(v) for v in person_box]
    return px1 <= wx <= px2 and py1 <= wy <= py2


def _compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def dedupe_people_boxes(people, overlap_iou=0.5):
    if not people:
        return []

    ordered = sorted(people, key=lambda item: float(item["score"]), reverse=True)
    kept = []

    for candidate in ordered:
        overlaps_existing = any(
            _compute_iou(candidate["box"], existing["box"]) >= overlap_iou for existing in kept
        )
        if not overlaps_existing:
            kept.append(candidate)

    return kept


def dedupe_detection_boxes(items, overlap_iou=0.5):
    if not items:
        return []

    ordered = sorted(items, key=lambda item: float(item["score"]), reverse=True)
    kept = []

    for candidate in ordered:
        overlaps_existing = any(
            _compute_iou(candidate["box"], existing["box"]) >= overlap_iou for existing in kept
        )
        if not overlaps_existing:
            kept.append(candidate)

    return kept


def split_people_and_weapons(boxes, scores, labels, min_score=THREAT_DECISION_THRESHOLD):
    people = []
    weapons = []

    if len(labels) == 0:
        return people, weapons

    for idx, label in enumerate(labels.astype(int)):
        score = float(scores[idx]) if len(scores) > idx else 0.0
        if score < min_score:
            continue

        item = {
            "index": idx,
            "box": boxes[idx],
            "score": score,
            "label": label,
        }

        if _is_person_label(label):
            people.append(item)
        elif _is_weapon_label(label):
            weapons.append(item)

    return people, weapons


def analyze_scene_relationships(boxes, scores, labels, min_score=THREAT_DECISION_THRESHOLD):
    people_boxes, weapon_boxes = split_people_and_weapons(boxes, scores, labels, min_score=min_score)

    # Clean, one-instance lists for people and weapons.
    people_boxes = [p for p in people_boxes if float(p.get("score", 0.0)) >= PERSON_CONFIDENCE_THRESHOLD]
    people_boxes = dedupe_people_boxes(people_boxes, overlap_iou=PERSON_NMS_IOU_THRESHOLD)
    weapon_boxes = [w for w in weapon_boxes if float(w.get("score", 0.0)) >= 0.80]
    weapon_boxes = dedupe_detection_boxes(weapon_boxes, overlap_iou=WEAPON_NMS_IOU_THRESHOLD)

    armed_person_indices = set()
    linked_weapon_indices = set()

    # Prioritize threat relation: weapon center inside person box.
    for w_idx, weapon in enumerate(weapon_boxes):
        for p_idx, person in enumerate(people_boxes):
            if is_weapon_center_inside_person(weapon["box"], person["box"]):
                armed_person_indices.add(p_idx)
                linked_weapon_indices.add(w_idx)

    armed_people = [people_boxes[p_idx] for p_idx in sorted(armed_person_indices)]
    person_only = [
        person for p_idx, person in enumerate(people_boxes) if p_idx not in armed_person_indices
    ]
    linked_weapon_boxes = [
        weapon for w_idx, weapon in enumerate(weapon_boxes) if w_idx in linked_weapon_indices
    ]
    weapon_only = [
        weapon for w_idx, weapon in enumerate(weapon_boxes) if w_idx not in linked_weapon_indices
    ]

    threat_detected = len(weapon_boxes) > 0
    confidence_score = 0.0
    if threat_detected:
        best_weapon = max(weapon_boxes, key=lambda item: float(item["score"]))
        confidence_score = float(best_weapon["score"])

    return {
        "people_boxes": people_boxes,
        "weapon_boxes": weapon_boxes,
        "armed_people": armed_people,
        "person_only": person_only,
        "linked_weapon_boxes": linked_weapon_boxes,
        "weapon_only": weapon_only,
        "armed_person_indices": sorted(list(armed_person_indices)),
        "threat_detected": threat_detected,
        "object_detected": False,
        "threat_type": "WEAPON" if threat_detected else None,
        "confidence_score": confidence_score,
        "threat_count": len(weapon_boxes),
        "object_count": 0,
        "status_message": "THREAT DETECTED" if threat_detected else "ALL CLEAR",
    }


def _draw_box_label(image, box, color, text, thickness=3, text_color=(255, 255, 255), fill_color=None):
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    line_len = int(min(img_w, img_h) * 0.05)
    cv2.line(image, (x1, y1), (x1 + line_len, y1), color, 2)
    cv2.line(image, (x1, y1), (x1, y1 + line_len), color, 2)
    cv2.line(image, (x2, y2), (x2 - line_len, y2), color, 2)
    cv2.line(image, (x2, y2), (x2, y2 - line_len), color, 2)

    (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    background = fill_color if fill_color is not None else color
    cv2.rectangle(image, (x1, y1 - 25), (x1 + text_width + 10, y1), background, -1)
    cv2.putText(
        image,
        text,
        (x1 + 5, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        text_color,
        2,
    )


def _draw_dashed_rect(image, box, color, thickness=2, dash_len=10, gap_len=6):
    x1, y1, x2, y2 = [int(v) for v in box]

    step = max(1, dash_len + gap_len)

    for pos in range(x1, x2, step):
        segment_end = min(pos + dash_len, x2)
        cv2.line(image, (pos, y1), (segment_end, y1), color, thickness)
        cv2.line(image, (pos, y2), (segment_end, y2), color, thickness)

    for pos in range(y1, y2, step):
        segment_end = min(pos + dash_len, y2)
        cv2.line(image, (x1, pos), (x1, segment_end), color, thickness)
        cv2.line(image, (x2, pos), (x2, segment_end), color, thickness)


def draw_tactical_boxes(image, scene):
    weapons = scene.get("weapon_boxes", [])
    weapon_threshold = 0.80

    # Weapon-only rendering: no person overlays and no yellow labels.
    for weapon in weapons:
        weapon_score = float(weapon.get("score", 0.0))
        if weapon_score < weapon_threshold:
            continue

        weapon_label = int(weapon.get("label", -1))
        raw_name = str(config.WEAPON_CLASSES.get(weapon_label, "WEAPON")).upper()
        display_name = "GUN" if raw_name in {"PISTOL", "FIREARM", "GUN"} else raw_name

        _draw_box_label(
            image,
            weapon["box"],
            (0, 0, 255),
            f"{display_name}: {int(weapon_score * 100)}%",
            thickness=2,
        )

    return image


def preprocess_for_live(frame):
    original_h, original_w = frame.shape[:2]
    resized = cv2.resize(frame, (LIVE_INFERENCE_SIZE, LIVE_INFERENCE_SIZE))
    scale_x = original_w / float(LIVE_INFERENCE_SIZE)
    scale_y = original_h / float(LIVE_INFERENCE_SIZE)
    return resized, scale_x, scale_y


def _scale_boxes_to_original(boxes, scale_x, scale_y):
    if len(boxes) == 0:
        return np.array([])
    scaled = boxes.copy().astype(np.float32)
    scaled[:, [0, 2]] *= scale_x
    scaled[:, [1, 3]] *= scale_y
    return scaled


def _filter_relevant_detections(boxes, scores, labels, min_score=THREAT_DECISION_THRESHOLD):
    if len(labels) == 0:
        return np.array([]), np.array([]), np.array([])

    keep_indices = []
    for idx, label in enumerate(labels.astype(int)):
        if (_is_person_label(label) or _is_weapon_label(label)) and float(scores[idx]) >= min_score:
            keep_indices.append(idx)

    if not keep_indices:
        return np.array([]), np.array([]), np.array([])

    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]


def _track_boxes_optical_flow(prev_gray, curr_gray, boxes, frame_shape):
    if prev_gray is None or len(boxes) == 0:
        return boxes, False

    points = []
    for x1, y1, x2, y2 in boxes:
        points.append([(float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0])
    prev_points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )

    if next_points is None or status is None:
        return boxes, False

    tracked_boxes = boxes.copy().astype(np.float32)
    valid_count = 0
    frame_h, frame_w = frame_shape[:2]

    for index, ok in enumerate(status.flatten()):
        if ok != 1:
            continue

        prev_pt = prev_points[index][0]
        next_pt = next_points[index][0]
        shift_x = float(next_pt[0] - prev_pt[0])
        shift_y = float(next_pt[1] - prev_pt[1])

        tracked_boxes[index, [0, 2]] += shift_x
        tracked_boxes[index, [1, 3]] += shift_y

        tracked_boxes[index, [0, 2]] = np.clip(tracked_boxes[index, [0, 2]], 0, frame_w - 1)
        tracked_boxes[index, [1, 3]] = np.clip(tracked_boxes[index, [1, 3]], 0, frame_h - 1)

        if tracked_boxes[index, 2] > tracked_boxes[index, 0] and tracked_boxes[index, 3] > tracked_boxes[index, 1]:
            valid_count += 1

    tracking_ok = valid_count > 0 and (valid_count / max(len(boxes), 1)) >= 0.5
    return tracked_boxes, tracking_ok


def infer_fused_detections(image, record_scan=True):
    if record_scan:
        now = datetime.now()
        with log_lock:
            scan_timestamps.append(now)
            if len(scan_timestamps) > MAX_LOG_ENTRIES:
                del scan_timestamps[:-MAX_LOG_ENTRIES]

    original_shape = image.shape

    results_a = faster_rcnn.predict(image)
    results_b = efficient_det.predict(image)
    predictions_list = [results_a, results_b]

    def unpack_predictions(prediction):
        if isinstance(prediction, dict):
            return (
                prediction.get("boxes", np.array([])),
                prediction.get("scores", np.array([])),
                prediction.get("labels", np.array([])),
            )
        if isinstance(prediction, (tuple, list)) and len(prediction) >= 3:
            return prediction[0], prediction[1], prediction[2]
        return np.array([]), np.array([]), np.array([])

    try:
        fused_boxes, fused_scores, fused_labels = ensemble_fusion.fuse_results(
            predictions_list, original_shape, weights=[1, 1]
        )
    except Exception as error:
        print(f"[ERROR] Fusion failed: {error}")
        # Fallback: use Faster R-CNN results so static image uploads still produce output.
        fused_boxes, fused_scores, fused_labels = unpack_predictions(results_a)

    # Extra fallback for defensive robustness when fusion returns empty unexpectedly.
    if len(fused_labels) == 0:
        fallback_boxes, fallback_scores, fallback_labels = unpack_predictions(results_a)
        if len(fallback_labels) > 0:
            fused_boxes, fused_scores, fused_labels = fallback_boxes, fallback_scores, fallback_labels

    if len(fused_labels) > 0:
        fused_boxes, fused_scores, fused_labels = _filter_relevant_detections(
            fused_boxes, fused_scores, fused_labels, min_score=THREAT_DECISION_THRESHOLD
        )

    return fused_boxes, fused_scores, fused_labels


def run_ensemble_inference(image):
    original_shape = image.shape
    original_h, original_w = original_shape[:2]

    resized_for_inference = cv2.resize(image, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
    fused_boxes, fused_scores, fused_labels = infer_fused_detections(resized_for_inference, record_scan=True)

    scaled_boxes = fused_boxes
    if len(fused_boxes) > 0:
        scale_x = original_w / float(INFERENCE_WIDTH)
        scale_y = original_h / float(INFERENCE_HEIGHT)
        scaled_boxes = fused_boxes.copy().astype(np.float32)
        scaled_boxes[:, [0, 2]] *= scale_x
        scaled_boxes[:, [1, 3]] *= scale_y

    scene = analyze_scene_relationships(
        scaled_boxes, fused_scores, fused_labels, min_score=THREAT_DECISION_THRESHOLD
    )

    processed_image = draw_tactical_boxes(image.copy(), scene)

    return (
        processed_image,
        scene["threat_detected"],
        scene["threat_count"],
        scene["threat_type"],
        scene["confidence_score"],
        scene["object_detected"],
        scene["object_count"],
    )


def append_detection_log(source, threat_type, confidence_score):
    entry = {
        "timestamp": datetime.now(),
        "source": source,
        "threat_type": threat_type,
        "confidence_score": round(float(confidence_score) * 100, 2),
    }
    with log_lock:
        detection_logs.append(entry)
        if len(detection_logs) > MAX_LOG_ENTRIES:
            del detection_logs[:-MAX_LOG_ENTRIES]


def is_allowed_file(filename):
    extension = os.path.splitext(filename.lower())[1]
    return extension in ALLOWED_IMAGE_EXTENSIONS or extension in ALLOWED_VIDEO_EXTENSIONS


def process_video_file(input_path, output_path):
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise RuntimeError("Unable to open uploaded video.")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 20.0

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    threat_found = False
    object_found = False
    best_threat_type = None
    best_confidence = 0.0
    frame_count = 0
    last_boxes = np.array([])
    last_scores = np.array([])
    last_labels = np.array([])

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame_count += 1

        run_inference_now = (frame_count % VIDEO_FRAME_SKIP == 0) or (len(last_labels) == 0)
        if run_inference_now:
            resized_frame = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
            resized_boxes, resized_scores, resized_labels = infer_fused_detections(resized_frame, record_scan=True)

            if len(resized_labels) > 0:
                scaled_boxes = resized_boxes.copy().astype(np.float32)
                scale_x = frame.shape[1] / float(INFERENCE_WIDTH)
                scale_y = frame.shape[0] / float(INFERENCE_HEIGHT)
                scaled_boxes[:, [0, 2]] *= scale_x
                scaled_boxes[:, [1, 3]] *= scale_y
                last_boxes = scaled_boxes
                last_scores = resized_scores
                last_labels = resized_labels
            else:
                last_boxes = np.array([])
                last_scores = np.array([])
                last_labels = np.array([])

        scene = analyze_scene_relationships(
            last_boxes, last_scores, last_labels, min_score=THREAT_DECISION_THRESHOLD
        )

        processed_frame = draw_tactical_boxes(frame.copy(), scene)

        frame_threat = scene["threat_detected"]
        frame_threat_type = scene["threat_type"]
        frame_confidence = scene["confidence_score"]
        frame_object = scene["object_detected"]

        threat_found = threat_found or frame_threat
        object_found = object_found or frame_object
        if frame_threat and frame_confidence > best_confidence:
            best_confidence = frame_confidence
            best_threat_type = frame_threat_type
        writer.write(processed_frame)

    capture.release()
    writer.release()
    return threat_found, object_found, best_threat_type, best_confidence


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detection")
def detection_page():
    return render_template("detection.html")


@app.route("/monitoring")
def monitoring_page():
    return render_template("monitoring.html")


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/alerts")
def alerts_page():
    with log_lock:
        alerts = [
            {
                "timestamp": item["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "source": item["source"],
                "threat_type": item["threat_type"],
                "confidence_score": item["confidence_score"],
            }
            for item in sorted(detection_logs, key=lambda x: x["timestamp"], reverse=True)
        ]
    return render_template("alerts.html", alerts=alerts)


@app.route("/analytics")
def analytics_page():
    now = datetime.now()
    days = [now.date() - timedelta(days=offset) for offset in range(6, -1, -1)]
    day_labels = [day.strftime("%a") for day in days]

    with log_lock:
        logs_snapshot = list(detection_logs)
        scans_snapshot = list(scan_timestamps)

    day_scan_counts = {day: 0 for day in days}
    day_threat_counts = {day: 0 for day in days}
    hourly_counts = [0] * 24
    threat_distribution = {}
    source_distribution = {"Uploaded Image": 0, "Uploaded Video": 0, "Live CCTV": 0}

    for scan_time in scans_snapshot:
        scan_day = scan_time.date()
        if scan_day in day_scan_counts:
            day_scan_counts[scan_day] += 1

    for log_item in logs_snapshot:
        log_time = log_item["timestamp"]
        log_day = log_time.date()
        if log_day in day_threat_counts:
            day_threat_counts[log_day] += 1

        hourly_counts[log_time.hour] += 1
        threat_name = log_item["threat_type"]
        threat_distribution[threat_name] = threat_distribution.get(threat_name, 0) + 1

        source_name = log_item["source"]
        if source_name not in source_distribution:
            source_distribution[source_name] = 0
        source_distribution[source_name] += 1

    total_scans = len(scans_snapshot)
    total_threats = len(logs_snapshot)
    safe_scans = max(total_scans - total_threats, 0)
    uptime_seconds = int((now - app_started_at).total_seconds())
    uptime_display = f"{(uptime_seconds // 3600):02d}:{((uptime_seconds % 3600) // 60):02d}:{(uptime_seconds % 60):02d}"

    analytics_data = {
        "trends": {
            "labels": day_labels,
            "detections": [day_scan_counts[day] for day in days],
            "threats": [day_threat_counts[day] for day in days],
        },
        "threatDist": {
            "labels": list(threat_distribution.keys()) if threat_distribution else ["No Threats"],
            "data": list(threat_distribution.values()) if threat_distribution else [1],
        },
        "sourceUsage": {
            "labels": list(source_distribution.keys()),
            "data": list(source_distribution.values()),
        },
        "hourly": hourly_counts,
    }

    return render_template(
        "analytics.html",
        total_scans=total_scans,
        total_threats=total_threats,
        safe_scans=safe_scans,
        uptime=uptime_display,
        analytics_data=analytics_data,
    )


@app.route("/crime-map")
@app.route("/crimemap")
def crime_map_page():
    return render_template("crimemap.html")


@app.route("/guide")
def guide_page():
    return render_template("guide.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file provided."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"status": "error", "message": "No selected file."}), 400

    filename = secure_filename(file.filename)
    if not is_allowed_file(filename):
        return jsonify({"status": "error", "message": "Unsupported file format."}), 400

    extension = os.path.splitext(filename.lower())[1]
    token = uuid4().hex
    uploaded_name = f"{token}{extension}"
    uploaded_path = os.path.join(UPLOAD_DIR, uploaded_name)
    file.save(uploaded_path)

    if extension in ALLOWED_IMAGE_EXTENSIONS:
        image = cv2.imread(uploaded_path)
        if image is None:
            return jsonify({"status": "error", "message": "Invalid image."}), 400

        (
            processed_image,
            threat_detected,
            threat_count,
            threat_type,
            confidence_score,
            object_detected,
            object_count,
        ) = run_ensemble_inference(image)
        output_name = f"{token}_processed.jpg"
        output_path = os.path.join(PROCESSED_DIR, output_name)
        cv2.imwrite(output_path, processed_image)

        if threat_detected:
            append_detection_log("Uploaded Image", threat_type or "Firearm", confidence_score)

        return jsonify(
            {
                "status": "success",
                "file_type": "image",
                "processed_url": url_for("static", filename=f"processed/{output_name}"),
                "original_url": url_for("static", filename=f"uploads/{uploaded_name}"),
                "threat": threat_detected,
                "status_message": "THREAT DETECTED" if threat_detected else "ALL CLEAR",
                "threat_count": threat_count,
                "threat_type": threat_type,
                "confidence_score": round(confidence_score * 100, 2),
                "object_detected": object_detected,
                "object_count": object_count,
            }
        )

    output_name = f"{token}_processed.mp4"
    output_path = os.path.join(PROCESSED_DIR, output_name)
    threat_detected, object_detected, threat_type, confidence_score = process_video_file(uploaded_path, output_path)

    if threat_detected:
        append_detection_log("Uploaded Video", threat_type or "Firearm", confidence_score)

    return jsonify(
        {
            "status": "success",
            "file_type": "video",
            "processed_url": url_for("static", filename=f"processed/{output_name}"),
            "original_url": url_for("static", filename=f"uploads/{uploaded_name}"),
            "threat": threat_detected,
            "status_message": "THREAT DETECTED" if threat_detected else "ALL CLEAR",
            "threat_count": 1 if threat_detected else 0,
            "threat_type": threat_type,
            "confidence_score": round(confidence_score * 100, 2),
            "object_detected": object_detected,
        }
    )


@app.route("/detect", methods=["POST"])
def detect_file():
    incoming_file = request.files.get("image") or request.files.get("file")
    if incoming_file is None:
        return jsonify({"status": "error", "message": "No image provided."}), 400

    if incoming_file.filename == "":
        return jsonify({"status": "error", "message": "No selected image."}), 400

    filename = secure_filename(incoming_file.filename)
    extension = os.path.splitext(filename.lower())[1]
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({"status": "error", "message": "Only image files are supported on /detect."}), 400

    token = uuid4().hex
    input_name = f"{token}_input{extension}"
    input_path = os.path.join(UPLOAD_DIR, input_name)
    incoming_file.save(input_path)

    image = cv2.imread(input_path)
    if image is None:
        return jsonify({"status": "error", "message": "Invalid image content."}), 400

    # Model inference happens here.
    (
        processed_image,
        threat_detected,
        threat_count,
        threat_type,
        confidence_score,
        object_detected,
        object_count,
    ) = run_ensemble_inference(image)

    output_name = f"{token}_output.jpg"
    output_path = os.path.join(UPLOAD_DIR, output_name)
    cv2.imwrite(output_path, processed_image)

    if threat_detected:
        append_detection_log("Uploaded Image", threat_type or "Firearm", confidence_score)

    return jsonify(
        {
            "status": "success",
            "processed_url": url_for("static", filename=f"uploads/{output_name}"),
            "threat": threat_detected,
            "status_message": "THREAT DETECTED" if threat_detected else "ALL CLEAR",
            "threat_count": threat_count,
            "threat_type": threat_type,
            "confidence_score": round(confidence_score * 100, 2),
            "object_detected": object_detected,
            "object_count": object_count,
        }
    )


def generate_camera_stream():
    global last_webcam_threat, last_webcam_confidence, last_webcam_threat_type, last_live_log_time

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return

    frame_index = 0
    frames_since_detection = LIVE_DETECTION_INTERVAL
    previous_gray = None
    tracked_boxes = np.array([])
    tracked_scores = np.array([])
    tracked_labels = np.array([])
    local_last_object = False
    local_last_threat_count = 0
    local_last_object_count = 0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            frame_index += 1
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            scene_changed = False
            if previous_gray is not None:
                frame_delta = cv2.absdiff(previous_gray, current_gray)
                scene_changed = float(np.mean(frame_delta)) > LIVE_SCENE_CHANGE_THRESHOLD

            run_full_detection = (
                frames_since_detection >= LIVE_DETECTION_INTERVAL
                or len(tracked_labels) == 0
                or scene_changed
            )

            if not run_full_detection and len(tracked_boxes) > 0 and previous_gray is not None:
                tracked_boxes, tracking_ok = _track_boxes_optical_flow(previous_gray, current_gray, tracked_boxes, frame.shape)
                if not tracking_ok:
                    run_full_detection = True

            if run_full_detection:
                resized_live_frame, scale_x, scale_y = preprocess_for_live(frame)
                live_boxes, live_scores, live_labels = infer_fused_detections(resized_live_frame, record_scan=True)
                live_boxes = _scale_boxes_to_original(live_boxes, scale_x, scale_y)
                tracked_boxes, tracked_scores, tracked_labels = _filter_relevant_detections(
                    live_boxes, live_scores, live_labels, min_score=THREAT_DECISION_THRESHOLD
                )
                frames_since_detection = 0
            else:
                frames_since_detection += 1

            previous_gray = current_gray

            scene = analyze_scene_relationships(
                tracked_boxes, tracked_scores, tracked_labels, min_score=THREAT_DECISION_THRESHOLD
            )

            processed_frame = draw_tactical_boxes(frame.copy(), scene)

            threat_detected = scene["threat_detected"]
            threat_type = scene["threat_type"]
            confidence_score = scene["confidence_score"]
            local_last_object = scene["object_detected"]
            local_last_threat_count = scene["threat_count"]
            local_last_object_count = scene["object_count"]

            with threat_lock:
                last_webcam_threat = threat_detected
                last_webcam_confidence = confidence_score
                last_webcam_threat_type = threat_type
                app.config["LAST_WEBCAM_OBJECT"] = local_last_object
                app.config["LAST_WEBCAM_THREAT_COUNT"] = local_last_threat_count
                app.config["LAST_WEBCAM_OBJECT_COUNT"] = local_last_object_count

            if threat_detected:
                now = datetime.now()
                if (
                    last_live_log_time is None
                    or (now - last_live_log_time).total_seconds() >= LIVE_LOG_INTERVAL_SECONDS
                ):
                    append_detection_log("Live CCTV", threat_type or "Firearm", confidence_score)
                    last_live_log_time = now

            success, buffer = cv2.imencode(".jpg", processed_frame)
            if not success:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.03)
    finally:
        camera.release()
        with threat_lock:
            last_webcam_threat = False
            last_webcam_confidence = 0.0
            last_webcam_threat_type = None
            app.config["LAST_WEBCAM_OBJECT"] = False
            app.config["LAST_WEBCAM_THREAT_COUNT"] = 0
            app.config["LAST_WEBCAM_OBJECT_COUNT"] = 0


@app.route("/video_feed")
def video_feed():
    return Response(generate_camera_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/threat_status")
def threat_status():
    with threat_lock:
        threat = last_webcam_threat
        threat_type = last_webcam_threat_type
        confidence = round(float(last_webcam_confidence) * 100, 2)
        object_detected = bool(app.config.get("LAST_WEBCAM_OBJECT", False))
        threat_count = int(app.config.get("LAST_WEBCAM_THREAT_COUNT", 0))
        object_count = int(app.config.get("LAST_WEBCAM_OBJECT_COUNT", 0))

    return jsonify(
        {
            "threat": bool(threat),
            "status_message": "THREAT DETECTED" if bool(threat) else "ALL CLEAR",
            "object": object_detected,
            "threat_type": threat_type,
            "confidence_score": confidence,
            "threat_count": threat_count,
            "object_count": object_count,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)