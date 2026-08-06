import os

import cv2


def get_video_info(video_path: str) -> dict:
    """Return basic metadata about a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_sec": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
        }
    finally:
        cap.release()
    return info


def xyxy_to_xywh(bbox: list) -> list:
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def xywh_to_xyxy(bbox: list) -> list:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
