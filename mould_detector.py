"""Auto-detect mould centroids from the video using the mould_detection model.

Samples a few frames near the start of the video, runs mould_detection.pt,
averages the bounding boxes, and returns centroids sorted left-to-right
as relative (x, y) fractions of frame size.
"""

import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

from config import (
    MOULD_DETECTION_MODEL_PATH,
    MOULD_DETECTION_CONFIDENCE,
    MOULD_DETECTION_SAMPLE_FRAMES,
    NUM_MOULDS,
    MOULD_CENTROIDS_FALLBACK,
)


def detect_mould_centroids(video_path):
    """Run mould detection on the first few frames and return stable centroids.

    Strategy:
      1. Sample MOULD_DETECTION_SAMPLE_FRAMES evenly across the first 10% of the video.
      2. For each frame, collect bounding boxes detected above confidence threshold.
      3. Cluster boxes by proximity (moulds are fixed, so boxes across frames
         for the same mould will be close together).
      4. Average each cluster to get a stable centroid.
      5. Sort clusters left-to-right → Mould 1, 2, 3.
      6. Return as relative (x, y) fractions.

    Falls back to MOULD_CENTROIDS_FALLBACK if detection fails.

    Returns:
        dict {mould_idx: (rel_cx, rel_cy)}  (0-based, sorted left to right)
    """
    try:
        model = YOLO(MOULD_DETECTION_MODEL_PATH)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Sample from first 10% of video (moulds don't move)
        sample_end = max(int(total_frames * 0.10), MOULD_DETECTION_SAMPLE_FRAMES * 10)
        sample_points = np.linspace(0, sample_end, MOULD_DETECTION_SAMPLE_FRAMES, dtype=int)

        # Collect all raw centroids across sampled frames
        raw_centroids = []  # list of (cx_px, cy_px)

        for fnum in sample_points:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fnum))
            ret, frame = cap.read()
            if not ret:
                continue

            results = model(frame, verbose=False, conf=MOULD_DETECTION_CONFIDENCE)
            boxes = results[0].boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                raw_centroids.append((cx, cy))

        cap.release()

        if not raw_centroids:
            print("[MouldDetector] No moulds detected — using fallback centroids.")
            return MOULD_CENTROIDS_FALLBACK

        # Cluster centroids by proximity using simple greedy merging
        # (works well since moulds are spatially separated and fixed)
        clusters = _cluster_centroids(raw_centroids, frame_w, frame_h)

        if len(clusters) != NUM_MOULDS:
            print(f"[MouldDetector] Expected {NUM_MOULDS} moulds, found {len(clusters)} clusters "
                  f"— using fallback centroids.")
            return MOULD_CENTROIDS_FALLBACK

        # Sort clusters left to right by x coordinate
        clusters_sorted = sorted(clusters, key=lambda c: c[0])

        # Convert to relative coordinates
        centroids = {}
        for idx, (cx_px, cy_px) in enumerate(clusters_sorted):
            centroids[idx] = (cx_px / frame_w, cy_px / frame_h)
            print(f"[MouldDetector] Mould {idx + 1}: centroid=({cx_px / frame_w:.3f}, {cy_px / frame_h:.3f})")

        return centroids

    except Exception as e:
        print(f"[MouldDetector] Error during detection: {e} — using fallback centroids.")
        return MOULD_CENTROIDS_FALLBACK


def _cluster_centroids(points, frame_w, frame_h, merge_threshold_frac=0.15):
    """Merge points that are within merge_threshold_frac * frame_width of each other.

    Returns list of (mean_cx, mean_cy) for each cluster.
    """
    threshold = merge_threshold_frac * frame_w
    clusters = []  # list of lists of (cx, cy)

    for cx, cy in points:
        matched = False
        for cluster in clusters:
            # Compare to cluster mean
            mean_cx = np.mean([p[0] for p in cluster])
            mean_cy = np.mean([p[1] for p in cluster])
            dist = np.sqrt((cx - mean_cx) ** 2 + (cy - mean_cy) ** 2)
            if dist < threshold:
                cluster.append((cx, cy))
                matched = True
                break
        if not matched:
            clusters.append([(cx, cy)])

    # Return mean of each cluster
    return [
        (np.mean([p[0] for p in c]), np.mean([p[1] for p in c]))
        for c in clusters
    ]
