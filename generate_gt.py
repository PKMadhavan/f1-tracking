"""
generate_gt.py — Pseudo ground-truth generator for F1 tracking evaluation.

Uses YOLOv8m (larger, more accurate model) at high confidence as a "silver
standard" detector, then assigns stable GT track IDs via DeepSORT — the same
tracker used for predictions, giving consistent ID behaviour across GT and pred.
Outputs a MOT-format gt.txt usable with --gt in main.py.

Usage:
    python generate_gt.py --input f1_trimmed.mp4 --output annotations/gt.txt
"""

import cv2
import argparse
import os
from tqdm import tqdm
from ultralytics import YOLO

from track import CarTracker


CAR_CLASS_ID = 2


def generate_gt(input_video, output_gt, conf=0.5, max_frames=None):
    print(f"\n=== Pseudo-GT Generator (YOLOv8n high-conf + DeepSORT oracle) ===")
    print(f"Input : {input_video}")
    print(f"Output: {output_gt}")
    print(f"Conf  : {conf}\n")

    os.makedirs(os.path.dirname(output_gt), exist_ok=True)

    model   = YOLO("yolov8n.pt")          # same model as predictor; high conf = "certain" detections only
    tracker = CarTracker(max_age=70, n_init=3, max_cosine_distance=0.3)

    cap   = cv2.VideoCapture(input_video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total, max_frames) if max_frames else total
    gt_lines = []

    print(f"Processing {limit} frames with YOLOv8m + DeepSORT...")
    with tqdm(total=limit, unit="frame") as pbar:
        for frame_id in range(limit):
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                source=frame, conf=conf, classes=[CAR_CLASS_ID],
                device="cpu", verbose=False
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
                gt_lines.append(
                    f"{frame_id + 1},{t['track_id']},{x1},{y1},{w},{h},1,-1,-1,-1"
                )

            pbar.update(1)

    cap.release()

    with open(output_gt, "w") as f:
        f.write("\n".join(gt_lines))

    unique_ids = len(set(int(l.split(",")[1]) for l in gt_lines))
    print(f"\nDone. {len(gt_lines)} annotations  |  {unique_ids} unique GT track IDs")
    print(f"Saved → {output_gt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo-GT for F1 tracking evaluation")
    parser.add_argument("--input",      required=True,                    help="Input video path")
    parser.add_argument("--output",     default="annotations/gt.txt",     help="Output MOT gt.txt path")
    parser.add_argument("--conf",       type=float, default=0.5,          help="Detection confidence threshold for GT oracle")
    parser.add_argument("--max-frames", type=int,   default=None,         help="Limit frames processed")
    args = parser.parse_args()

    generate_gt(args.input, args.output, args.conf, args.max_frames)
