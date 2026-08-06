"""
analysis.py — Generate and save all analysis plots + sample frames.

Run after the tracking pipeline to produce:
  - track_analysis.png   (cars-per-frame timeline + track-length histogram)
  - sample_frames.png    (5 annotated frames spread across the video)

Usage:
    f1-analyze --input f1_trimmed.mp4
"""

import argparse
import logging

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from f1_tracking.detect import CarDetector
from f1_tracking.evaluate import compute_track_stability, count_basic_stats
from f1_tracking.logging_config import setup_logging
from f1_tracking.track import CarTracker
from f1_tracking.utils import get_video_info
from f1_tracking.visualize import draw_tracks

logger = logging.getLogger(__name__)


def load_tracked_data(video_path: str, conf: float, max_frames: int | None = None):
    """Re-run pipeline to collect pred_frames and annotated_frames."""
    detector = CarDetector(conf_threshold=conf)
    tracker = CarTracker(max_age=70, n_init=5, max_cosine_distance=0.3)

    info = get_video_info(video_path)
    fps = info["fps"]
    total = info["total_frames"]
    limit = min(total, max_frames) if max_frames else total

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    pred_frames = {}
    annotated_frames = []

    try:
        for frame_id in tqdm(range(limit), desc="Loading tracking data", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break
            dets = detector.detect(frame)
            tracks = tracker.update(dets, frame)
            pred_frames[frame_id] = tracks
            annotated_frames.append(draw_tracks(frame, tracks))
    finally:
        cap.release()

    return pred_frames, annotated_frames, fps


def save_track_analysis(pred_frames: dict, fps: float):
    frame_ids = sorted(pred_frames.keys())
    cars_per_frame = [len(pred_frames[f]) for f in frame_ids]
    times = [f / fps for f in frame_ids]
    stats = count_basic_stats(pred_frames)
    stability = compute_track_stability(pred_frames)

    track_map: dict = {}
    for tracks in pred_frames.values():
        for t in tracks:
            track_map[t["track_id"]] = track_map.get(t["track_id"], 0) + 1
    lengths = list(track_map.values())

    _fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].fill_between(times, cars_per_frame, alpha=0.35, color="royalblue")
    axes[0].plot(times, cars_per_frame, color="royalblue", linewidth=0.9)
    axes[0].axhline(
        stats["avg_simultaneous_cars"],
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"avg = {stats['avg_simultaneous_cars']} cars/frame",
    )
    axes[0].set_xlabel("Time (s)", fontsize=11)
    axes[0].set_ylabel("Cars detected", fontsize=11)
    axes[0].set_title("Cars Detected Per Frame Over Time", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(lengths, bins=min(30, len(lengths)) or 1, color="darkorange", edgecolor="white", alpha=0.85)
    axes[1].axvline(
        stability["avg_track_length"],
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"avg = {stability['avg_track_length']} frames",
    )
    axes[1].set_xlabel("Track length (frames)", fontsize=11)
    axes[1].set_ylabel("Number of tracks", fontsize=11)
    axes[1].set_title(
        f"Track Length Distribution  —  {stats['total_unique_tracks']} unique IDs  "
        f"|  fragmentation = {stability['fragmentation']}",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("track_analysis.png", dpi=150)
    plt.close()
    logger.info("Saved → track_analysis.png")
    return stats, stability


def save_sample_frames(annotated_frames: list, pred_frames: dict, fps: float) -> None:
    n = len(annotated_frames)
    sample_ids = [int(n * p) for p in [0.05, 0.25, 0.50, 0.75, 0.95]]

    _fig, axes = plt.subplots(1, len(sample_ids), figsize=(22, 4))
    for ax, fid in zip(axes, sample_ids):
        rgb = cv2.cvtColor(annotated_frames[fid], cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        n_cars = len(pred_frames.get(fid, []))
        ax.set_title(f"t={fid/fps:.1f}s\n{n_cars} car{'s' if n_cars != 1 else ''}", fontsize=10)
        ax.axis("off")

    plt.suptitle("Sample Annotated Frames — YOLOv8n + DeepSORT", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("sample_frames.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → sample_frames.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    pred_frames, annotated_frames, fps = load_tracked_data(args.input, args.conf, args.max_frames)
    stats, stability = save_track_analysis(pred_frames, fps)
    save_sample_frames(annotated_frames, pred_frames, fps)

    logger.info("=== Summary ===")
    for k, v in {**stats, **stability}.items():
        logger.info("%-28s: %s", k, v)


if __name__ == "__main__":
    main()
