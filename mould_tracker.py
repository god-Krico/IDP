"""Track mould-wise progress by checking if mould centroids fall inside detection boxes."""

from collections import defaultdict, deque

from config import (
    NUM_MOULDS, CLASS_NAMES, MIN_STAGE_DURATION_SECONDS,
    MOULD_CENTROIDS_FALLBACK, CONFIRM_WINDOW_SIZE, CONFIRM_MIN_HITS,
)


class MouldAssigner:
    """Assigns detections to moulds using centroid-in-box logic.

    Each mould has a fixed centroid position (relative coordinates).
    A YOLO detection is assigned to a mould if that mould's centroid
    falls inside the detection's bounding box. This handles oblique
    camera angles correctly.

    Centroids can be passed in at construction time (from auto-detection)
    or will fall back to MOULD_CENTROIDS_FALLBACK from config.
    """

    def __init__(self, frame_width, frame_height, centroids=None):
        """
        Args:
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.
            centroids: dict {mould_idx: (rel_x, rel_y)} of relative centroid positions.
                       If None, falls back to MOULD_CENTROIDS_FALLBACK from config.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        source = centroids if centroids is not None else MOULD_CENTROIDS_FALLBACK
        # Convert relative centroids to absolute pixel coordinates
        self.centroids_px = {}
        for mould_idx, (rx, ry) in source.items():
            self.centroids_px[mould_idx] = (rx * frame_width, ry * frame_height)

    def assign(self, x1, y1, x2, y2):
        """Return list of mould indices whose centroid falls inside bbox (x1,y1,x2,y2).

        Returns list because a large bbox could theoretically contain multiple moulds.
        """
        matches = []
        for mould_idx, (cx, cy) in self.centroids_px.items():
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                matches.append(mould_idx)
        return matches


class DetectionFilter:
    """Filters spurious detections using a sliding window confirmation.

    For each mould, keeps a window of the last N frames' detections.
    A class is only "confirmed" if it appears in at least K of those N frames.
    """

    def __init__(self, num_moulds=NUM_MOULDS,
                 window_size=CONFIRM_WINDOW_SIZE,
                 min_hits=CONFIRM_MIN_HITS):
        self.window_size = window_size
        self.min_hits = min_hits
        # Per-mould sliding window of class_ids (one entry per processed frame, None if no detection)
        self._windows = {i: deque(maxlen=window_size) for i in range(num_moulds)}
        # Track the last confirmed class per mould to avoid duplicate logging
        self._last_confirmed = {i: None for i in range(num_moulds)}

    def push(self, mould_idx, class_id):
        """Push a detection for this mould in the current frame.

        Args:
            mould_idx: Which mould.
            class_id: Detected class, or None if nothing detected for this mould.
        """
        self._windows[mould_idx].append(class_id)

    def push_empty(self, mould_idx):
        """Record that nothing was detected for this mould in the current frame."""
        self._windows[mould_idx].append(None)

    def is_confirmed(self, mould_idx, class_id):
        """Check if class_id has enough hits in the recent window for this mould."""
        window = self._windows[mould_idx]
        count = sum(1 for c in window if c == class_id)
        return count >= self.min_hits


class ProgressTracker:
    """Accumulates per-mould stage observations and builds a timeline."""

    def __init__(self, num_moulds=NUM_MOULDS):
        self.num_moulds = num_moulds
        # raw_observations[mould_idx] = [(timestamp_str, class_id, confidence), ...]
        self.raw_observations = defaultdict(list)
        self.detection_filter = DetectionFilter(num_moulds)
        # Track what was detected per mould in the current frame (before confirmation)
        self._current_frame_detections = {}

    def begin_frame(self):
        """Call at the start of each processed frame to reset per-frame state."""
        self._current_frame_detections = {}

    def record_detection(self, mould_idx, class_id):
        """Record a raw detection for this frame (before confirmation)."""
        self._current_frame_detections[mould_idx] = class_id

    def end_frame(self, timestamp_str, confidence_map=None):
        """Call after all detections for a frame are recorded.

        Pushes detections into the sliding window, checks confirmation,
        and logs confirmed observations.

        Args:
            timestamp_str: The timestamp for this frame.
            confidence_map: Optional dict {mould_idx: confidence}.
        """
        if confidence_map is None:
            confidence_map = {}

        confirmed_detections = []

        for mould_idx in range(self.num_moulds):
            cls_id = self._current_frame_detections.get(mould_idx, None)

            if cls_id is not None:
                self.detection_filter.push(mould_idx, cls_id)
            else:
                self.detection_filter.push_empty(mould_idx)

            # Check if any class is now confirmed for this mould
            if cls_id is not None and self.detection_filter.is_confirmed(mould_idx, cls_id):
                conf = confidence_map.get(mould_idx, 0.0)
                self.raw_observations[mould_idx].append((timestamp_str, cls_id, conf))
                confirmed_detections.append({
                    "mould": mould_idx,
                    "class": CLASS_NAMES[cls_id],
                    "confidence": conf,
                })

        return confirmed_detections

    def get_latest_stages(self):
        """Return the most recent confirmed stage for each mould (for live display)."""
        latest = {}
        for mould_idx in range(self.num_moulds):
            obs = self.raw_observations.get(mould_idx, [])
            if obs:
                latest[mould_idx] = CLASS_NAMES[obs[-1][1]]
            else:
                latest[mould_idx] = None
        return latest

    def build_timeline(self):
        """Build a cleaned timeline for each mould.

        Returns dict: mould_idx -> list of {stage, start, end} dicts.
        """
        from timestamp_ocr import timestamp_to_seconds

        timelines = {}
        for mould_idx in range(self.num_moulds):
            obs = self.raw_observations.get(mould_idx, [])
            if not obs:
                timelines[mould_idx] = []
                continue

            # Sort by timestamp
            obs_sorted = sorted(obs, key=lambda x: timestamp_to_seconds(x[0]))

            # Build raw segments: consecutive runs of the same class
            segments = []
            current_class = obs_sorted[0][1]
            current_start = obs_sorted[0][0]
            current_end = obs_sorted[0][0]

            for ts, cls_id, conf in obs_sorted[1:]:
                if cls_id == current_class:
                    current_end = ts
                else:
                    segments.append({
                        "stage": CLASS_NAMES[current_class],
                        "class_id": current_class,
                        "start": current_start,
                        "end": current_end,
                    })
                    current_class = cls_id
                    current_start = ts
                    current_end = ts

            # Last segment
            segments.append({
                "stage": CLASS_NAMES[current_class],
                "class_id": current_class,
                "start": current_start,
                "end": current_end,
            })

            # Filter out very short segments (noise)
            filtered = []
            for seg in segments:
                duration = timestamp_to_seconds(seg["end"]) - timestamp_to_seconds(seg["start"])
                if duration >= MIN_STAGE_DURATION_SECONDS:
                    filtered.append(seg)

            # Merge adjacent segments of the same stage (after noise removal)
            merged = []
            for seg in filtered:
                if merged and merged[-1]["stage"] == seg["stage"]:
                    merged[-1]["end"] = seg["end"]
                else:
                    merged.append(seg)

            timelines[mould_idx] = merged

        return timelines
