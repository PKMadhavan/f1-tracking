"""CLI entry point for the tracking pipeline (console script: f1-track)."""

import argparse

from f1_tracking.logging_config import setup_logging
from f1_tracking.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="F1 Car Detection & Tracking")
    parser.add_argument("--input", required=True, help="Path to input F1 video")
    parser.add_argument("--output", default="output_tracked.mp4", help="Output video path")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model variant")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames (for testing)")
    parser.add_argument("--gt", default=None, help="Ground truth .txt (MOT format) for MOTA/IDF1")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    run_pipeline(
        input_video=args.input,
        output_video=args.output,
        model=args.model,
        conf=args.conf,
        max_frames=args.max_frames,
        gt_path=args.gt,
    )


if __name__ == "__main__":
    main()
