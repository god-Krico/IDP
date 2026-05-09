"""Productivity analysis processor — Streamlit-compatible adaptation of productivity.py.

Extracted from the Jupyter notebook prototype. No OpenCV windows, no video writing.
Takes boundary points from Streamlit's drawable canvas instead of an interactive UI.
All charts are returned as Plotly figures or numpy arrays for st.image().
"""

import cv2
import numpy as np
from collections import deque, Counter

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from ultralytics import YOLO


# ── Constants ───────────────────────────────────────────────────────────────

LEAN_CATEGORY_MAP = {
    "rebar_cage_placement": "VA",
    "concrete_pouring": "VA",
    "surface_finishing": "VA",
    "vacuum_lifting": "VA",
    "mould_cleaning": "NVAN",
    "Active (Physical Labor)": "NVAN",
    "Active (Unclassified Phase)": "NVAN",
    "idle": "NVA",
    "curing": "EXCLUDE",
}

PHASE_COLORS = {
    "idle": "#d3d3d3",
    "mould_cleaning": "#1f77b4",
    "rebar_cage_placement": "#ff7f0e",
    "concrete_pouring": "#2ca02c",
    "surface_finishing": "#9467bd",
    "curing": "#8c564b",
    "vacuum_lifting": "#e377c2",
    "Active (Physical Labor)": "#17becf",
    "Active (Unclassified Phase)": "#bcbd22",
}

LEAN_COLORS = {"VA": "#2ca02c", "NVAN": "#ff7f0e", "NVA": "#d62728"}

_PADDING_KERNEL = np.ones((301, 151), np.uint8)

MIN_WORKERS_PER_PHASE = {
    "mould_cleaning": 1,
    "rebar_cage_placement": 0,  # machine-assisted phase  — worker count doesn't gate it
    "concrete_pouring": 1,
    "surface_finishing": 1,
    "curing": 0,
    "vacuum_lifting": 0,   # machine-assisted phase  — worker count doesn't gate it
}
DEFAULT_MIN_WORKERS = 1
PERSON_CLASS_ID = [1]
CONFIRM_WINDOW_SIZE = 10
CONFIRM_MIN_HITS = 3
STATIC_FRAME_LIMIT = 5
STATIC_TOLERANCE = 15
MEMORY_LIMIT = 3

WRIST_MOVEMENT_THRESHOLD = 8
WRIST_HISTORY_LEN        = 5
KEYPOINT_CONF_MIN        = 0.4
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


# ── Lazy model loading ───────────────────────────────────────────────────────
# Models are loaded once per Python session and reused across calls.

_mould_model  = None
_phase_model  = None
_worker_model = None
_pose_model   = None


def load_models():
    """Load productivity models (lazy, called once)."""
    global _mould_model, _phase_model, _worker_model, _pose_model
    if _mould_model is None:
        _mould_model  = YOLO("model/finetuned_multi.pt")
        _phase_model  = YOLO("model/phase_detection.pt")
        _worker_model = YOLO("model/proximity_best.pt")
        _pose_model   = YOLO("yolov8n-pose.pt")   # auto-downloads on first run
    return _mould_model, _phase_model, _worker_model, _pose_model


# ── Boundary helpers ─────────────────────────────────────────────────────────

def get_boundary_y(cx, pts):
    """Interpolate boundary Y at a given X from the sorted polyline points."""
    if not pts or len(pts) < 2:
        return 0
    sorted_pts = sorted(pts, key=lambda p: p[0])
    xs = [p[0] for p in sorted_pts]
    ys = [p[1] for p in sorted_pts]
    return int(np.interp(cx, xs, ys))


def _iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    return inter / float((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


# ── Main processing function ─────────────────────────────────────────────────

def process_productivity(
    video_path,
    boundary_points=None,
    frame_skip=6,
    timelapse_interval=5,
    frame_callback=None,
):
    """
    Process a video for productivity analytics.

    Args:
        video_path:        Path to input video file.
        boundary_points:   List of (x, y) tuples defining the boundary polyline in
                           full-frame pixel coordinates. Workers whose feet (y2) are
                           ABOVE the interpolated boundary Y at their X position are
                           ignored. Pass None or [] to process the full frame.
        frame_skip:        Process every Nth frame (matches timelapse cadence).
        timelapse_interval: Real-world seconds represented by one timelapse frame.
        frame_callback:    Called after each processed frame with a state dict:
                               frame_idx, total_frames, annotated_frame,
                               timeline_history, confirmed_phases, workers_in_zone,
                               active_workers, ghost_count, blacklist_count

    Returns:
        dict with keys:
            timeline_history    {mould_id: [phase_str, ...]}
            heatmap_data        float32 numpy accumulator (frame_height x frame_width)
            worker_trajectories {worker_id: [(cx, cy), ...]}
            first_frame         BGR numpy array of frame 0
            frame_count         total frames read
            timelapse_interval
            frame_skip
    """
    mould_model, phase_model, worker_model, pose_model = load_models()

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Cannot read video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    use_boundary = bool(boundary_points and len(boundary_points) >= 2)

    # ── Detect mould zones ONCE before the main loop ────────────────────────
    # Moulds are physically fixed throughout the shoot.  Running detection once
    # on an early frame and reusing those dilated masks for the whole video
    # prevents mould-ID drift: without this, a frame where the middle mould is
    # momentarily occluded would cause the right mould to be labelled as mould 1.
    exclusive_zones: list = []

    def _build_zones_from_frame(src_frame):
        """Return list of exclusive binary masks, sorted left-to-right."""
        res = mould_model.predict(src_frame, imgsz=640, verbose=False)
        if res[0].masks is None:
            return []
        masks_raw = res[0].masks.data.cpu().numpy()
        boxes_raw = res[0].boxes.xyxy.cpu().numpy()
        order     = np.argsort(boxes_raw[:, 0])      # left → right
        masks_raw = masks_raw[order][:3]             # at most 3 moulds

        dilated_list = []
        for m in masks_raw:
            m_full = cv2.resize(m, (frame_width, frame_height))
            dilated_list.append(cv2.dilate(m_full, _PADDING_KERNEL, iterations=1))

        overlap = (np.sum(dilated_list, axis=0) > 1).astype(np.uint8)
        zones   = []
        for d in dilated_list:
            zones.append(cv2.bitwise_and(d, d, mask=cv2.bitwise_not(overlap)))
        return zones

    # Try the first frame, then up to 4 more evenly-spaced frames if needed
    for _sample_idx in range(5):
        _sample_pos = int(_sample_idx * total_frames / 10)
        cap.set(cv2.CAP_PROP_POS_FRAMES, _sample_pos)
        _ok, _sframe = cap.read()
        if _ok:
            exclusive_zones = _build_zones_from_frame(_sframe)
            if len(exclusive_zones) >= 2:
                break  # good enough

    if not exclusive_zones:
        # Fallback: process without zone constraints (phase/worker detection still runs)
        pass

    # Initialise per-mould state dictionaries now that we know the zone count
    timeline_history = {}
    phase_buffers    = {}
    confirmed_phases = {}
    for i in range(len(exclusive_zones)):
        timeline_history[i] = []
        phase_buffers[i]    = deque(maxlen=CONFIRM_WINDOW_SIZE)
        confirmed_phases[i] = "idle"

    # Reset cap to the beginning before the main processing loop
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── State ────────────────────────────────────────────────────────────────
    heatmap_data        = np.zeros((frame_height, frame_width), dtype=np.float32)
    worker_trajectories = {}
    frame_count         = 0
    active_worker_memory = {}
    prev_gray            = None
    blacklisted_ids      = set()
    false_positive_areas = []
    keypoint_history     = {}

    # ── Main loop ────────────────────────────────────────────────────────────
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            break
        if frame_count % frame_skip != 0:
            continue

        annotated = frame.copy()
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        # Draw boundary polyline on the frame for visual reference
        if use_boundary:
            pts_sorted = sorted(boundary_points, key=lambda p: p[0])
            for i in range(1, len(pts_sorted)):
                cv2.line(annotated, pts_sorted[i - 1], pts_sorted[i], (0, 255, 255), 2)
            for pt in pts_sorted:
                cv2.circle(annotated, pt, 5, (0, 200, 200), -1)

        # ── A. APPLY PRE-COMPUTED MOULD ZONES ───────────────────────────────
        # Zones were fixed once before the loop — IDs are stable for the whole video.
        tint = np.array([255, 150, 0], dtype=np.float32)
        for excl in exclusive_zones:
            annotated[excl > 0] = (
                annotated[excl > 0].astype(np.float32) * 0.7 + tint * 0.3
            ).astype(np.uint8)

        # ── B. PHASE DETECTION ───────────────────────────────────────────────
        raw_phases = {i: "idle" for i in range(len(exclusive_zones))}
        phase_res  = phase_model.predict(frame, imgsz=640, verbose=False)

        if phase_res[0].boxes is not None:
            for box, cls_id in zip(
                phase_res[0].boxes.xyxy.cpu().numpy(),
                phase_res[0].boxes.cls.cpu().numpy(),
            ):
                name = phase_model.names[int(cls_id)]
                px1, py1, px2, py2 = map(int, box)
                cx_p, cy_p = (px1 + px2) // 2, (py1 + py2) // 2
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(annotated, f"Phase: {name}", (px1, max(20, py1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                for i, zone in enumerate(exclusive_zones):
                    if zone[cy_p, cx_p] > 0:
                        raw_phases[i] = name
                        break

        for i in range(len(exclusive_zones)):
            phase_buffers[i].append(raw_phases[i])
            if Counter(phase_buffers[i])[raw_phases[i]] >= CONFIRM_MIN_HITS:
                confirmed_phases[i] = raw_phases[i]

        # ── C. WORKER TRACKING ───────────────────────────────────────────────
        zone_counts = {i: 0 for i in range(len(exclusive_zones))}
        w_res = worker_model.track(
            frame, classes=PERSON_CLASS_ID, persist=True,
            imgsz=640, tracker="bytetrack.yaml", conf=0.5, verbose=False,
        )
        current_ids = []

        if w_res[0].boxes.id is not None:
            for box, w_id in zip(
                w_res[0].boxes.xyxy.cpu().numpy(),
                w_res[0].boxes.id.cpu().numpy(),
            ):
                w_id = int(w_id)
                if w_id in blacklisted_ids:
                    continue
                x1, y1, x2, y2 = map(int, box)
                if (x2 - x1) > 250 or (y2 - y1) > 350:
                    continue
                cx, cy = (x1 + x2) // 2, y2  # feet tracking

                # Boundary check using interpolated polyline Y
                if use_boundary and cy < get_boundary_y(cx, boundary_points):
                    continue

                current_ids.append(w_id)
                if w_id not in active_worker_memory:
                    active_worker_memory[w_id] = {
                        "missed_frames": 0,
                        "is_moving": False,
                        "history": deque(maxlen=STATIC_FRAME_LIMIT),
                        "box": (x1, y1, x2, y2),
                        "centroid": (cx, cy),
                    }

                # ID-agnostic spatial motion detection
                is_moving = False
                if prev_gray is not None:
                    ry1, ry2 = max(0, y1), min(frame_height, y2)
                    rx1, rx2 = max(0, x1), min(frame_width, x2)
                    if ry2 > ry1 and rx2 > rx1:
                        diff = cv2.absdiff(
                            gray[ry1:ry2, rx1:rx2], prev_gray[ry1:ry2, rx1:rx2]
                        )
                        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                        is_moving = (
                            np.count_nonzero(thresh) / (thresh.size + 1e-6) * 100 > 5.0
                        )

                active_worker_memory[w_id].update({
                    "box": (x1, y1, x2, y2),
                    "centroid": (cx, cy),
                    "is_moving": is_moving,
                    "missed_frames": 0,
                })
                active_worker_memory[w_id]["history"].append((x1, y1, x2, y2, False))

        # Ghost aging
        to_delete = []
        for w_id in list(active_worker_memory.keys()):
            if w_id not in current_ids:
                active_worker_memory[w_id]["missed_frames"] += 1
                active_worker_memory[w_id]["is_moving"] = False
                lb = active_worker_memory[w_id]["box"]
                active_worker_memory[w_id]["history"].append((*lb, True))
                if active_worker_memory[w_id]["missed_frames"] > MEMORY_LIMIT:
                    to_delete.append(w_id)

        # Static artifact filter (5-frame jitter + spatial memory bank)
        to_blacklist = []
        for w_id, data in active_worker_memory.items():
            hist = list(data["history"])
            if len(hist) < 3:
                continue
            x_sp = max(h[0] for h in hist) - min(h[0] for h in hist)
            y_sp = max(h[1] for h in hist) - min(h[1] for h in hist)
            w_sp = max(h[2]-h[0] for h in hist) - min(h[2]-h[0] for h in hist)
            h_sp = max(h[3]-h[1] for h in hist) - min(h[3]-h[1] for h in hist)
            bl = False

            if len(hist) == STATIC_FRAME_LIMIT and max(x_sp, y_sp, w_sp, h_sp) <= STATIC_TOLERANCE:
                bl = True
                false_positive_areas.append(
                    tuple(sum(h[j] for h in hist) / len(hist) for j in range(4))
                )

            if not bl and len(hist) >= 3:
                l3 = hist[-3:]
                l3_sp = [
                    max(h[0] for h in l3) - min(h[0] for h in l3),
                    max(h[1] for h in l3) - min(h[1] for h in l3),
                    max(h[2]-h[0] for h in l3) - min(h[2]-h[0] for h in l3),
                    max(h[3]-h[1] for h in l3) - min(h[3]-h[1] for h in l3),
                ]
                if max(l3_sp) <= STATIC_TOLERANCE:
                    if (sum(1 for h in l3 if not h[4]) >= 1
                            and sum(1 for h in l3 if h[4]) >= 2):
                        for bad in false_positive_areas:
                            if _iou(data["box"], bad) > 0.65:
                                bl = True
                                break
            if bl:
                blacklisted_ids.add(w_id)
                to_blacklist.append(w_id)

        for w_id in set(to_blacklist + to_delete):
            active_worker_memory.pop(w_id, None)
            if w_id in current_ids:
                current_ids.remove(w_id)

        # Geometric deduplication
        valid_workers = []
        for w_id, data in active_worker_memory.items():
            x1, y1, x2, y2 = data["box"]
            cx, cy = data["centroid"]
            dup = False
            for idx, vw in enumerate(valid_workers):
                vx1, vy1, vx2, vy2 = vw["box"]
                vcx, vcy = vw["centroid"]
                if (np.hypot(cx - vcx, cy - vcy) < 25
                        or (vx1 <= cx <= vx2 and vy1 <= cy <= vy2)
                        or (x1 <= vcx <= x2 and y1 <= vcy <= y2)):
                    dup = True
                    if (x2-x1)*(y2-y1) > (vx2-vx1)*(vy2-vy1):
                        valid_workers[idx] = {
                            "id": w_id, "box": (x1, y1, x2, y2),
                            "centroid": (cx, cy), "missed": data["missed_frames"],
                            "is_moving": data["is_moving"],
                        }
                    break
            if not dup:
                valid_workers.append({
                    "id": w_id, "box": (x1, y1, x2, y2),
                    "centroid": (cx, cy), "missed": data["missed_frames"],
                    "is_moving": data["is_moving"],
                })

        # ── E. POSE-BASED WRIST ACTIVITY ────────────────────────────────────
        pose_res = pose_model.predict(frame, imgsz=640, conf=0.4, verbose=False)

        if pose_res[0].keypoints is not None:
            kp_all   = pose_res[0].keypoints.data.cpu().numpy()   # (N, 17, 3)
            kp_boxes = pose_res[0].boxes.xyxy.cpu().numpy()

            for kps, kp_box in zip(kp_all, kp_boxes):
                # Draw skeleton (cyan lines + magenta dots)
                for (ia, ib) in SKELETON_CONNECTIONS:
                    xa, ya, ca = kps[ia]
                    xb, yb, cb = kps[ib]
                    if ca > KEYPOINT_CONF_MIN and cb > KEYPOINT_CONF_MIN:
                        cv2.line(annotated,
                                 (int(xa), int(ya)), (int(xb), int(yb)),
                                 (0, 255, 255), 1)
                for (kx, ky, kc) in kps:
                    if kc > KEYPOINT_CONF_MIN:
                        cv2.circle(annotated, (int(kx), int(ky)), 3, (255, 0, 255), -1)

                # Match pose detection to a valid_worker by IoU
                best_iou, matched_idx = 0.3, -1
                for vidx, vw in enumerate(valid_workers):
                    vx1, vy1, vx2, vy2 = vw["box"]
                    iou = _iou(
                        (kp_box[0], kp_box[1], kp_box[2], kp_box[3]),
                        (vx1, vy1, vx2, vy2),
                    )
                    if iou > best_iou:
                        best_iou, matched_idx = iou, vidx

                if matched_idx < 0:
                    continue

                w_id = valid_workers[matched_idx]["id"]
                lx, ly, lc = kps[9]    # left wrist
                rx, ry, rc = kps[10]   # right wrist

                if w_id not in keypoint_history:
                    keypoint_history[w_id] = []

                if lc > KEYPOINT_CONF_MIN or rc > KEYPOINT_CONF_MIN:
                    keypoint_history[w_id].append(
                        (np.array([lx, ly]), np.array([rx, ry]), lc, rc)
                    )
                    if len(keypoint_history[w_id]) > WRIST_HISTORY_LEN:
                        keypoint_history[w_id].pop(0)

                hand_active = False
                if len(keypoint_history[w_id]) >= 2:
                    pl, pr, plc, prc = keypoint_history[w_id][-2]
                    cl, cr, clc, crc = keypoint_history[w_id][-1]
                    lw_move = (np.linalg.norm(cl - pl)
                               if plc > KEYPOINT_CONF_MIN and clc > KEYPOINT_CONF_MIN else 0)
                    rw_move = (np.linalg.norm(cr - pr)
                               if prc > KEYPOINT_CONF_MIN and crc > KEYPOINT_CONF_MIN else 0)
                    hand_active = (lw_move > WRIST_MOVEMENT_THRESHOLD
                                   or rw_move > WRIST_MOVEMENT_THRESHOLD)

                valid_workers[matched_idx]["hand_active"] = hand_active

        # Update heatmap, trajectories, and zone counts
        for vw in valid_workers:
            cx, cy = vw["centroid"]
            x1, y1, x2, y2 = vw["box"]
            w_id = vw["id"]

            if 0 <= cy < frame_height and 0 <= cx < frame_width:
                heatmap_data[cy, cx] += 1
            worker_trajectories.setdefault(w_id, []).append((cx, cy))

            hand_active = vw.get("hand_active", False)

            status = "idle"
            for i, zone in enumerate(exclusive_zones):
                if zone[cy, cx] > 0:
                    zone_counts[i] += 1
                    p = confirmed_phases.get(i, "idle")
                    if p == "idle":
                        status = (
                            "Active (Physical Labor)"
                            if (vw["is_moving"] or hand_active)
                            else "Active (Unclassified Phase)"
                        )
                    else:
                        status = p
                    break

            is_physically_active = vw["is_moving"] or hand_active
            is_ghost = vw["missed"] > 0
            color = (
                (150, 150, 150) if is_ghost
                else ((0, 165, 255) if (status != "idle" or is_physically_active) else (0, 0, 255))
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            if is_ghost:
                label = f"ID:{w_id} [GHOST]"
            elif hand_active and status == "idle":
                label = f"ID:{w_id} [Active-Hands]"
            else:
                label = f"ID:{w_id} [{status[:12]}]"

            cv2.putText(
                annotated, label, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

        # ── D. LABOR GATING ──────────────────────────────────────────────────
        for i in range(len(exclusive_zones)):
            phase   = confirmed_phases.get(i, "idle")
            workers = zone_counts[i]
            req     = MIN_WORKERS_PER_PHASE.get(phase, DEFAULT_MIN_WORKERS)
            if phase == "curing":
                state = "curing"
            elif phase in ("rebar_cage_placement", "vacuum_lifting") or \
                    raw_phases.get(i) in ("rebar_cage_placement", "vacuum_lifting"):
                # Machine-assisted phases (aerolifter): active for every frame the phase
                # model detects either — bypasses the temporal smoother and worker-count gate.
                state = phase if phase in ("rebar_cage_placement", "vacuum_lifting") \
                    else raw_phases.get(i)
            elif workers >= req:
                state = phase if phase != "idle" else "Active (Unclassified Phase)"
            else:
                state = "idle"
            timeline_history[i].append(state)

        prev_gray = gray.copy()

        if frame_callback:
            ghost_count = sum(
                1 for d in active_worker_memory.values() if d["missed_frames"] > 0
            )
            frame_callback({
                "frame_idx":      frame_count,
                "total_frames":   total_frames,
                "annotated_frame": annotated,
                "timeline_history": timeline_history,
                "confirmed_phases": {
                    i: confirmed_phases.get(i, "idle")
                    for i in range(len(exclusive_zones))
                },
                "workers_in_zone":  dict(zone_counts),
                "active_workers":   len(valid_workers),
                "ghost_count":      ghost_count,
                "blacklist_count":  len(blacklisted_ids),
            })

    cap.release()
    return {
        "timeline_history":    timeline_history,
        "heatmap_data":        heatmap_data,
        "worker_trajectories": worker_trajectories,
        "first_frame":         first_frame,
        "frame_count":         frame_count,
        "timelapse_interval":  timelapse_interval,
        "frame_skip":          frame_skip,
    }


# ── Chart builders ────────────────────────────────────────────────────────────

def build_crew_balance_chart(timeline_history, frame_skip, timelapse_interval):
    """Plotly Gantt of crew phases over real-world time."""
    rows = []
    for mould_id, history in timeline_history.items():
        if not history:
            continue
        cur_phase, start_idx = history[0], 0
        for fi, phase in enumerate(history):
            if phase != cur_phase or fi == len(history) - 1:
                start_min = (start_idx * frame_skip * timelapse_interval) / 60.0
                dur_min   = ((fi - start_idx) * frame_skip * timelapse_interval) / 60.0
                if dur_min > 0:
                    rows.append({
                        "Mould":          f"Mould {mould_id}",
                        "Phase":          cur_phase,
                        "Start (min)":    start_min,
                        "Duration (min)": round(dur_min, 2),
                    })
                cur_phase, start_idx = phase, fi

    if not rows:
        return None

    df   = pd.DataFrame(rows)
    base = pd.Timestamp("2026-01-01")
    df["Start"] = df["Start (min)"].apply(lambda m: base + pd.Timedelta(minutes=m))
    df["End"]   = df.apply(
        lambda r: base + pd.Timedelta(minutes=r["Start (min)"] + r["Duration (min)"]),
        axis=1,
    )
    fig = px.timeline(
        df, x_start="Start", x_end="End", y="Mould", color="Phase",
        color_discrete_map=PHASE_COLORS,
        hover_data=["Duration (min)"],
        title="Crew Balance Timeline (Real-World Minutes)",
    )
    fig.update_layout(
        xaxis_title="Real-World Time (min)", yaxis_title="",
        height=280, margin=dict(t=40, b=30),
        xaxis=dict(tickformat="%M:%S"),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def build_lean_pie_charts(timeline_history):
    """Plotly donut charts — one per mould — showing VA / NVAN / NVA split."""
    moulds = list(timeline_history.keys())
    if not moulds:
        return None
    n   = len(moulds)
    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{"type": "pie"}] * n],
        subplot_titles=[f"Mould {mid}" for mid in moulds],
    )
    for col_i, mould_id in enumerate(moulds, 1):
        counts = {"VA": 0, "NVAN": 0, "NVA": 0}
        for phase in timeline_history[mould_id]:
            cat = LEAN_CATEGORY_MAP.get(phase, "NVA")
            if cat != "EXCLUDE":
                counts[cat] += 1
        labels = [k for k, v in counts.items() if v > 0]
        values = [counts[k] for k in labels]
        colors = [LEAN_COLORS[k] for k in labels]
        if sum(values) > 0:
            fig.add_trace(
                go.Pie(
                    labels=labels, values=values, marker_colors=colors,
                    name=f"Mould {mould_id}", textinfo="percent+label", hole=0.35,
                ),
                row=1, col=col_i,
            )
    fig.update_layout(
        title="Lean Activity Distribution (VA / NVAN / NVA) — Curing Excluded",
        height=360, margin=dict(t=60, b=20),
    )
    return fig


def build_lean_bar_chart(timeline_history):
    """Live-updating 100% stacked bar showing running lean ratio per mould."""
    rows = []
    for mould_id, history in timeline_history.items():
        counts = {"VA": 0, "NVAN": 0, "NVA": 0}
        for phase in history:
            cat = LEAN_CATEGORY_MAP.get(phase, "NVA")
            if cat != "EXCLUDE":
                counts[cat] += 1
        total = sum(counts.values()) or 1
        for cat, cnt in counts.items():
            rows.append({
                "Mould":    f"Mould {mould_id}",
                "Category": cat,
                "Percent":  round(cnt / total * 100, 1),
            })
    if not rows:
        return None
    df  = pd.DataFrame(rows)
    fig = px.bar(
        df, x="Mould", y="Percent", color="Category",
        color_discrete_map=LEAN_COLORS, barmode="stack",
        title="Running Lean Ratio (% of Tracked Time)",
        text="Percent",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
    fig.update_layout(
        height=300, margin=dict(t=40, b=30),
        yaxis=dict(range=[0, 100], title="Percentage"),
        xaxis_title="",
        legend_title="Category",
    )
    return fig


def build_heatmap_overlay(heatmap_data, first_frame):
    """Blend motion heatmap onto first frame. Returns RGB numpy array for st.image."""
    blurred = cv2.GaussianBlur(heatmap_data, (99, 99), 0)
    norm    = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    mask    = (norm > 5).astype(np.float32)[:, :, np.newaxis]
    overlay = (first_frame * (1 - mask * 0.6) + colored * (mask * 0.6)).astype(np.uint8)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def build_spaghetti_diagram(worker_trajectories, first_frame):
    """Draw worker movement paths on the first frame. Returns RGB numpy array."""
    canvas = first_frame.copy()
    palette = [
        (255, 80,  80),  (80,  160, 255), (80,  255, 120), (255, 200, 0),
        (220, 80,  255), (0,   220, 220), (255, 120, 0),   (120, 255, 0),
    ]
    for idx, (w_id, path) in enumerate(worker_trajectories.items()):
        if len(path) < 2:
            continue
        color = palette[idx % len(palette)]
        for i in range(1, len(path)):
            cv2.line(canvas, path[i - 1], path[i], color, 1)
        cv2.circle(canvas, path[0],  5, color,        -1)   # start dot
        cv2.circle(canvas, path[-1], 5, (255, 255, 255), -1) # end dot (white)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def compute_lean_totals(timeline_history):
    """Return (overall_totals, per_mould) dicts of VA/NVAN/NVA frame counts."""
    totals    = {"VA": 0, "NVAN": 0, "NVA": 0}
    per_mould = {}
    for mould_id, history in timeline_history.items():
        counts = {"VA": 0, "NVAN": 0, "NVA": 0}
        for phase in history:
            cat = LEAN_CATEGORY_MAP.get(phase, "NVA")
            if cat != "EXCLUDE":
                counts[cat] += 1
                totals[cat] += 1
        per_mould[mould_id] = counts
    return totals, per_mould
