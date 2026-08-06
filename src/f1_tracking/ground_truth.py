"""
ground_truth.py — Pseudo ground-truth generator for F1 tracking evaluation.

Uses YOLOv8m (larger, more accurate model) at high confidence as a "silver
standard" detector, then assigns stable GT track IDs via DeepSORT — the same
tracker used for predictions, giving consistent ID behaviour across GT and pred.

Using a stronger model than the predictor (which defaults to yolov8n) is what
makes this GT independent of the predictor — an oracle built from the same
model it's evaluating would trivially agree with itself. Outputs a MOT-format
gt.txt usable with --gt in the tracking CLI.

Usage:
    f1-gt --input f1_trimmed.mp4 --output annotations/gt.txt
"""

import argparse
import logging
import os

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from f1_tracking.logging_config import setup_logging
from f1_tracking.track import CarTracker

logger = logging.getLogger(__name__)

CAR_CLASS_ID = 2
ORACLE_MODEL = "yolov8m.pt"


def generate_gt(input_video: str, output_gt: str, conf: float = 0.5, max_frames: int | None = None) -> None:
    logger.info("=== Pseudo-GT Generator (%s high-conf + DeepSORT oracle) ===", ORACLE_MODEL)
    logger.info("Input : %s", input_video)
    logger.info("Output: %s", output_gt)
    logger.info("Conf  : %s", conf)

    out_dir = os.path.dirname(output_gt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Deliberately a stronger model than the yolov8n predictor default, so the
    # pseudo-GT is not just the predictor agreeing with itself.
    model = YOLO(ORACLE_MODEL)
    tracker = CarTracker(max_age=70, n_init=3, max_cosine_distance=0.3)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total, max_frames) if max_frames else total
    gt_lines = []

    logger.info("Processing %d frames with %s + DeepSORT...", limit, ORACLE_MODEL)
    try:
        with tqdm(total=limit, unit="frame") as pbar:
            for frame_id in range(limit):
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(
                    source=frame,
                    conf=conf,
                    classes=[CAR_CLASS_ID],
                    device="cpu",
                    verbose=False,
                )
                dets = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    c = float(box.conf[0].cpu())
                    dets.append({"bbox": [x1, y1, x2, y2], "conf": c, "class_id": CAR_CLASS_ID})

                tracks = tracker.update(dets, frame)

                for t in tracks:
                    x1, y1, x2, y2 = t["bbox"]
                    w, h = x2 - x1, y2 - y1
                    gt_lines.append(f"{frame_id + 1},{t['track_id']},{x1},{y1},{w},{h},1,-1,-1,-1")

                pbar.update(1)
    finally:
        cap.release()

    with open(output_gt, "w") as f:
        f.write("\n".join(gt_lines))

    unique_ids = len({int(line.split(",")[1]) for line in gt_lines})
    logger.info("Done. %d annotations  |  %d unique GT track IDs", len(gt_lines), unique_ids)
    logger.info("Saved → %s", output_gt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pseudo-GT for F1 tracking evaluation")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="annotations/gt.txt", help="Output MOT gt.txt path")
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Detection confidence threshold for GT oracle"
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    generate_gt(args.input, args.output, args.conf, args.max_frames)


if __name__ == "__main__":
    main()
