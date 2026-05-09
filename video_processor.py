"""Main video processing pipeline: reads video, runs YOLO + OCR, tracks moulds."""

import cv2
import json
import os
from ultralytics import YOLO

from config import (
    MODEL_PATH,
    CLASS_NAMES,
    FRAME_SAMPLE_INTERVAL,
    DETECTION_CONFIDENCE_THRESHOLD,
    TIMESTAMP_REGION,
    NUM_MOULDS,
)
from timestamp_ocr import read_timestamp, seconds_to_timestamp
from mould_tracker import MouldAssigner, ProgressTracker
from mould_detector import detect_mould_centroids


def process_video(video_path, output_dir="output", frame_callback=None):
    """Process a video file and generate mould progress data.

    Args:
        video_path: Path to the input video.
        output_dir: Directory to save results.
        frame_callback: Optional callable(info_dict) called after each processed frame.
            info_dict contains: frame_idx, total_frames, timestamp, frame_bgr,
            detections, tracker, annotated_frame.

    Returns:
        dict with keys: timelines, metadata
    """
    video_path = str(video_path)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Auto-detect mould centroids using mould_detection.pt
    print("[VideoProcessor] Detecting mould positions...")
    centroids = detect_mould_centroids(video_path)

    # Load stage classification model
    model = YOLO(MODEL_PATH)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize mould assigner with auto-detected centroids
    assigner = MouldAssigner(frame_width, frame_height, centroids=centroids)
    tracker = ProgressTracker(NUM_MOULDS)

    frame_idx = 0
    processed_count = 0
    ocr_failures = 0
    last_date = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SAMPLE_INTERVAL == 0:
            # --- OCR: read timestamp and date from frame ---
            timestamp, date_str = read_timestamp(frame, TIMESTAMP_REGION)
            if date_str is not None:
                last_date = date_str
            if timestamp is None:
                ocr_failures += 1
                elapsed_sec = int(frame_idx / fps)
                timestamp = seconds_to_timestamp(elapsed_sec)

            # --- YOLO: detect stages ---
            results = model(frame, verbose=False, conf=DETECTION_CONFIDENCE_THRESHOLD)

            annotated_frame = results[0].plot() if results else frame

            # Tell tracker a new frame is starting
            tracker.begin_frame()
            confidence_map = {}

            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                    # Find which mould centroids fall inside this bbox
                    mould_matches = assigner.assign(x1, y1, x2, y2)

                    for mould_idx in mould_matches:
                        tracker.record_detection(mould_idx, cls_id)
                        confidence_map[mould_idx] = conf

            # Commit frame — applies sliding window filter, returns confirmed detections only
            frame_detections = tracker.end_frame(timestamp, confidence_map)

            processed_count += 1

            # Send real-time update to dashboard
            if frame_callback:
                frame_callback({
                    "frame_idx": frame_idx,
                    "total_frames": total_frames,
                    "processed_count": processed_count,
                    "timestamp": timestamp,
                    "date": last_date,
                    "detections": frame_detections,
                    "tracker": tracker,
                    "annotated_frame": annotated_frame,
                    "ocr_failures": ocr_failures,
                })

        frame_idx += 1

    cap.release()

    # Build final timelines
    timelines = tracker.build_timeline()

    summary = {}
    for mould_idx in range(NUM_MOULDS):
        mould_name = f"Mould {mould_idx + 1}"
        stages = timelines.get(mould_idx, [])
        summary[mould_name] = [
            {"stage": s["stage"], "start": s["start"], "end": s["end"]}
            for s in stages
        ]

    metadata = {
        "video_path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "frames_processed": processed_count,
        "resolution": f"{frame_width}x{frame_height}",
        "ocr_failures": ocr_failures,
        "sample_interval": FRAME_SAMPLE_INTERVAL,
        "date": last_date,
        "mould_centroids": {str(k): list(v) for k, v in centroids.items()},
    }

    result = {"timelines": summary, "metadata": metadata}
    output_path = os.path.join(output_dir, "progress_report.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    import sys

    video = sys.argv[1] if len(sys.argv) > 1 else "video/TLC00001.MP4"
    result = process_video(video)

    print("\n" + "=" * 60)
    print("MOULD PROGRESS REPORT")
    print("=" * 60)
    for mould_name, stages in result["timelines"].items():
        print(f"\n{mould_name}:")
        if not stages:
            print("  No activity detected.")
        for s in stages:
            print(f"  {s['stage']:25s}  {s['start']} -> {s['end']}")
