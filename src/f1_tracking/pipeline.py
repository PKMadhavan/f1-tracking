"""Core end-to-end tracking pipeline, extracted for reuse by the CLI and the Streamlit app."""

import logging

import cv2
from tqdm import tqdm

from f1_tracking.detect import CarDetector
from f1_tracking.evaluate import (
    compute_metrics_with_gt,
    compute_track_stability,
    count_basic_stats,
    load_mot_gt,
)
from f1_tracking.track import CarTracker
from f1_tracking.utils import get_video_info
from f1_tracking.visualize import draw_tracks, write_video

logger = logging.getLogger(__name__)


def run_pipeline(
    input_video: str,
    output_video: str = "output_tracked.mp4",
    model: str = "yolov8n.pt",
    conf: float = 0.25,
    max_frames: int | None = None,
    gt_path: str | None = None,
    progress_callback=None,
) -> dict:
    """
    Run the full detect → track → annotate → evaluate pipeline.

    Args:
        progress_callback: optional callable(frame_id, total_frames) invoked per frame,
                            used by the Streamlit app to drive a progress bar.

    Returns:
        Dict with keys: stats, stability, gt_metrics (None if no gt_path), output_video.
    """
    logger.info("=== F1 Car Detection & Tracking Pipeline ===")
    logger.info("Input  : %s", input_video)
    logger.info("Output : %s", output_video)
    logger.info("Model  : %s  |  Conf threshold: %s", model, conf)

    info = get_video_info(input_video)
    fps = info["fps"]
    total = info["total_frames"]
    logger.info(
        "Video  : %dx%d  %d frames @ %.1f fps  (%.1fs)",
        info["width"],
        info["height"],
        total,
        fps,
        info["duration_sec"],
    )

    detector = CarDetector(model_name=model, conf_threshold=conf)
    tracker = CarTracker(max_age=70, n_init=5, max_cosine_distance=0.3)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_video}")

    annotated_frames = []
    pred_frames = {}
    frame_id = 0
    limit = min(total, max_frames) if max_frames else total

    logger.info("Processing %d frames...", limit)
    try:
        with tqdm(total=limit, unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or (max_frames and frame_id >= max_frames):
                    break

                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame)

                pred_frames[frame_id] = tracks
                annotated_frames.append(draw_tracks(frame, tracks))

                frame_id += 1
                pbar.update(1)
                if progress_callback:
                    progress_callback(frame_id, limit)
    finally:
        cap.release()

    write_video(annotated_frames, output_video, fps)

    stats = count_basic_stats(pred_frames)
    stability = compute_track_stability(pred_frames)

    logger.info("=== Results ===")
    logger.info("Frames processed       : %s", stats["total_frames"])
    logger.info("Total detections       : %s", stats["total_detections"])
    logger.info("Total unique track IDs : %s", stats["total_unique_tracks"])
    logger.info("Max cars in one frame  : %s", stats["max_simultaneous_cars"])
    logger.info("Avg cars per frame     : %s", stats["avg_simultaneous_cars"])

    logger.info("=== Track Stability (no GT required) ===")
    logger.info("Avg track length       : %s frames", stability["avg_track_length"])
    logger.info("Fragmentation ratio    : %s  (ideal ~= 1.0)", stability["fragmentation"])
    logger.info("Short tracks (<10f)    : %.0f%% of all tracks", stability["short_track_ratio"] * 100)
    logger.info(
        "Detection rate         : %.0f%% of frames have detections", stability["detection_rate"] * 100
    )

    gt_metrics = None
    if gt_path:
        logger.info("Loading ground truth from %s ...", gt_path)
        gt_frames = load_mot_gt(gt_path)
        gt_metrics = compute_metrics_with_gt(gt_frames, pred_frames)
        logger.info("=== MOT Metrics ===")
        logger.info("MOTA        : %.2f%%", gt_metrics["MOTA"])
        logger.info("IDF1        : %.2f%%", gt_metrics["IDF1"])
        logger.info("ID Switches : %s", gt_metrics["ID_Switches"])
        logger.info("Misses      : %s", gt_metrics["Misses"])
        logger.info("False Pos.  : %s", gt_metrics["FP"])

    logger.info("Done. Output: %s", output_video)

    return {
        "stats": stats,
        "stability": stability,
        "gt_metrics": gt_metrics,
        "output_video": output_video,
    }
