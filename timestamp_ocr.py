"""OCR module to extract timestamps from video frames."""

import re
import cv2
import numpy as np

# Lazy-initialize the reader (GPU if available, else CPU)
_reader = None

def _import_easyocr():
    global easyocr
    try:
        import easyocr as _easyocr
        easyocr = _easyocr
    except ImportError:
        raise ImportError("easyocr is required for OCR features. Install it with: pip install easyocr")
_last_crop_hash = None
_last_timestamp = None
_last_date = None


def _get_reader():
    global _reader
    if _reader is None:
        _import_easyocr()
        _reader = easyocr.Reader(["en"], gpu=True)
    return _reader


def extract_timestamp_region(frame, region):
    """Crop the timestamp region from a video frame.

    Args:
        frame: BGR numpy array from OpenCV.
        region: Tuple (x1_frac, y1_frac, x2_frac, y2_frac) as fractions of frame size.
    """
    h, w = frame.shape[:2]
    x1 = int(w * region[0])
    y1 = int(h * region[1])
    x2 = int(w * region[2])
    y2 = int(h * region[3])
    return frame[y1:y2, x1:x2]


def preprocess_for_ocr(crop):
    """Preprocess timestamp crop for better OCR accuracy."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Upscale for better OCR
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Adaptive threshold to handle varying backgrounds
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresh


# Common timestamp patterns
_TIMESTAMP_PATTERNS = [
    # HH:MM:SS format
    r"(\d{1,2}):(\d{2}):(\d{2})",
    # YYYY-MM-DD HH:MM:SS
    r"\d{4}[-/]\d{2}[-/]\d{2}\s+(\d{1,2}):(\d{2}):(\d{2})",
    # DD/MM/YYYY HH:MM:SS
    r"\d{2}[-/]\d{2}[-/]\d{4}\s+(\d{1,2}):(\d{2}):(\d{2})",
]

# Date pattern: MM/DD/YYYY as seen in the video overlay
_DATE_PATTERN = r"(\d{2})/(\d{2})/(\d{4})"


def parse_timestamp_text(text):
    """Parse OCR text to extract a timestamp string.

    Returns the time as "HH:MM:SS" or None if not found.
    """
    text = text.strip().replace("\n", " ")
    for pattern in _TIMESTAMP_PATTERNS:
        match = re.search(pattern, text)
        if match:
            h, m, s = match.group(1), match.group(2), match.group(3)
            return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    return None


def parse_date_text(text):
    """Parse OCR text to extract date from MM/DD/YYYY format.

    Returns date as "DD MM YYYY" or None if not found.
    """
    text = text.strip().replace("\n", " ")
    match = re.search(_DATE_PATTERN, text)
    if match:
        mm, dd, yyyy = match.group(1), match.group(2), match.group(3)
        return f"{dd} {mm} {yyyy}"
    return None


def read_timestamp(frame, region):
    """Read timestamp and date from the given frame region.

    Caches result when the crop hasn't changed (timestamp updates ~1/sec,
    so consecutive frames within the same second return cached value).

    Returns (time_str "HH:MM:SS", date_str "DD MM YYYY") tuple.
    Either value can be None if not found.
    """
    global _last_crop_hash, _last_timestamp, _last_date

    crop = extract_timestamp_region(frame, region)
    if crop.size == 0:
        return None, None

    # Quick pixel hash to detect if timestamp region changed
    crop_small = cv2.resize(crop, (64, 16))
    crop_hash = crop_small.mean()
    if _last_crop_hash is not None and abs(crop_hash - _last_crop_hash) < 0.5 and _last_timestamp:
        return _last_timestamp, _last_date

    # Feed raw crop directly — preprocessing destroys colons
    reader = _get_reader()
    results = reader.readtext(crop, detail=0, paragraph=False)
    full_text = " ".join(results)
    ts = parse_timestamp_text(full_text)
    dt = parse_date_text(full_text)

    if ts is not None:
        _last_crop_hash = crop_hash
        _last_timestamp = ts
    if dt is not None:
        _last_date = dt

    return ts, dt


def timestamp_to_seconds(ts_str):
    """Convert 'HH:MM:SS' to total seconds."""
    parts = ts_str.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def seconds_to_timestamp(total_seconds):
    """Convert total seconds to 'HH:MM:SS'."""
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
