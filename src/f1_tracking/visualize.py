import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _color_for_id(track_id: int) -> tuple:
    """Return a deterministic BGR color for a given track ID."""
    np.random.seed(int(track_id) * 37)
    return tuple(int(x) for x in np.random.randint(80, 255, 3))


def draw_tracks(frame: np.ndarray, tracks: list[dict]) -> np.ndarray:
    """
    Overlay bounding boxes and track IDs onto a single frame.

    Args:
        frame:  BGR frame from cv2.
        tracks: Output of CarTracker.update() — list of track dicts.

    Returns:
        Annotated copy of the frame (original is not modified).
    """
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        tid = t["track_id"]
        color = _color_for_id(tid)

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label: colored background + white text
        label = f"Car #{tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def write_video(frames: list[np.ndarray], output_path: str, fps: float) -> None:
    """
    Write a list of annotated BGR frames to an MP4 file.

    Tries H.264 ("avc1") first, since that's the only codec HTML5 <video>
    tags (browsers, Streamlit's st.video) reliably play back. "mp4v"
    (MPEG-4 Part 2) writes a valid MP4 that OpenCV/ffmpeg can read but that
    Chrome/Safari refuse to decode, which shows up as a video player stuck
    at 0:00 with no visible frame. Falls back to "mp4v" only if the
    environment's OpenCV/ffmpeg build can't encode H.264.

    Args:
        frames:      List of np.ndarray frames (all same size).
        output_path: Output file path, e.g. 'output_tracked.mp4'.
        fps:         Frame rate copied from the source video.
    """
    if not frames:
        raise ValueError("No frames to write.")

    h, w = frames[0].shape[:2]

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    if not writer.isOpened():
        logger.warning("H.264 (avc1) encoder unavailable; falling back to mp4v (not browser-playable)")
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise OSError(f"Could not open video writer for: {output_path}")

    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.release()

    logger.info("Video saved → %s (%d frames @ %.1f fps)", output_path, len(frames), fps)
