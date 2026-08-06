"""
Quickstart — run the F1 tracking pipeline programmatically and inspect the results.

This is the library-level equivalent of the `f1-track` CLI command, useful as a
starting point for embedding the pipeline in your own scripts/notebooks.

Usage:
    python examples/quickstart.py --input f1_clip.mp4 --max-frames 200
"""

import argparse
import logging

from f1_tracking.logging_config import setup_logging
from f1_tracking.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to an F1 race video")
    parser.add_argument("--output", default="quickstart_output.mp4", help="Annotated output video path")
    parser.add_argument(
        "--max-frames", type=int, default=200, help="Frames to process (keep small for a quick run)"
    )
    args = parser.parse_args()

    setup_logging(verbose=False)
    logger = logging.getLogger("quickstart")

    result = run_pipeline(
        input_video=args.input,
        output_video=args.output,
        model="yolov8n.pt",
        conf=0.25,
        max_frames=args.max_frames,
    )

    logger.info("Annotated video written to %s", result["output_video"])
    logger.info("Detection stats: %s", result["stats"])
    logger.info("Track stability: %s", result["stability"])


if __name__ == "__main__":
    main()
