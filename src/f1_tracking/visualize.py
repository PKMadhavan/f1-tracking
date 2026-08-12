import logging
import shutil
import subprocess

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
    Write a list of annotated BGR frames to an MP4 file, encoded as H.264 —
    the only codec HTML5 <video> tags (browsers, Streamlit's st.video)
    reliably play back.

    OpenCV's bundled ffmpeg (in the opencv-python-headless PyPI wheel) is a
    static build without libx264, since that codec's licensing keeps it out
    of redistributable wheels — cv2.VideoWriter(..., "avc1", ...) silently
    can't open on a stock install. So this pipes raw frames to the *system*
    ffmpeg binary instead (present via Homebrew locally, apt in Docker/CI),
    which does bundle libx264. "mp4v" (MPEG-4 Part 2) is avoided entirely:
    it produces a valid MP4 that OpenCV can read back but that Chrome/Safari
    refuse to decode, which shows up as a video player stuck at 0:00 with no
    visible frame. Falls back to cv2's mp4v writer only if no system ffmpeg
    binary is on PATH at all.

    Args:
        frames:      List of np.ndarray frames (all same size).
        output_path: Output file path, e.g. 'output_tracked.mp4'.
        fps:         Frame rate copied from the source video.
    """
    if not frames:
        raise ValueError("No frames to write.")

    h, w = frames[0].shape[:2]

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for f in frames:
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read()
        if proc.wait() != 0:
            raise OSError(f"ffmpeg failed writing {output_path}: {stderr.decode(errors='replace')}")
    else:
        logger.warning("System ffmpeg not found on PATH; falling back to mp4v (not browser-playable)")
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            raise OSError(f"Could not open video writer for: {output_path}")
        try:
            for f in frames:
                writer.write(f)
        finally:
            writer.release()

    logger.info("Video saved → %s (%d frames @ %.1f fps)", output_path, len(frames), fps)
