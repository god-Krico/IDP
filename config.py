"""Configuration for the mould progress tracking system."""

# YOLO Models
MODEL_PATH = "model/progress.pt"               # Stage classification model
MOULD_DETECTION_MODEL_PATH = "model/mould_detection.pt"  # Mould locator model

# Class definitions
CLASS_NAMES = {
    0: "Mould Cleaning",
    1: "Rebar Cage Placement",
    2: "Concrete Pouring",
    3: "Surface Finishing",
    4: "Curing",
    5: "Vacuum Lifting",
}

# Stage colors for visualization (plotly-compatible)
STAGE_COLORS = {
    "Mould Cleaning": "#3498db",
    "Rebar Cage Placement": "#d00000",
    "Concrete Pouring": "#4600b0",
    "Surface Finishing": "#2ecc71",
    "Curing": "#9b59b6",
    "Vacuum Lifting": "#b2b900",
}

# Number of moulds in the casting yard
NUM_MOULDS = 3

# Mould centroid positions — used as FALLBACK if auto-detection fails.
# Auto-detection runs mould_detection.pt on the first few frames and derives
# centroids from the actual bounding boxes (sorted left to right).
# Override these only if the mould detection model is unavailable.
MOULD_CENTROIDS_FALLBACK = {
    0: (0.1,   0.22),    # Mould 1 (leftmost)
    1: (0.397, 0.254),   # Mould 2 (centre)
    2: (0.735, 0.358),   # Mould 3 (rightmost)
}

# How many frames to sample at the start of the video to establish mould centroids
MOULD_DETECTION_SAMPLE_FRAMES = 10
# Minimum detection confidence for the mould locator model
MOULD_DETECTION_CONFIDENCE = 0.5

# Video processing
FRAME_SAMPLE_INTERVAL = 1  # Process every Nth frame (at 30fps = ~1 per second)
DETECTION_CONFIDENCE_THRESHOLD = 0.5

# Timestamp OCR region (relative coordinates: x1, y1, x2, y2 as fractions of frame size)
# Bottom strip of the video where timestamp is displayed - adjust as needed
TIMESTAMP_REGION = (0.448, 0.97, 0.6, 1.0)

# Minimum duration (in seconds) for a stage to be considered valid (filters noise)
MIN_STAGE_DURATION_SECONDS = 30

# Detection confirmation filter:
# A detection is only logged if the same class is seen for a mould in at least
# CONFIRM_MIN_HITS of the last CONFIRM_WINDOW_SIZE processed frames.
# This filters out spurious single-frame false positives.
CONFIRM_WINDOW_SIZE = 10    # look at the last N frames
CONFIRM_MIN_HITS = 3        # must appear in at least this many of those N frames
