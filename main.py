import cv2
import argparse
from tqdm import tqdm

from detect import CarDetector
from track import CarTracker
from visualize import draw_tracks, write_video
from evaluate import count_basic_stats, compute_metrics_with_gt, compute_track_stability, load_mot_gt
from utils import get_video_info


def run_pipeline(
    input_video: str,
    output_video: str = "output_tracked.mp4",
    model: str = "yolov8n.pt",
    conf: float = 0.25,
    max_frames: int = None,
    gt_path: str = None,
):
    print("\n=== F1 Car Detection & Tracking Pipeline ===")
    print(f"Input  : {input_video}")
    print(f"Output : {output_video}")
    print(f"Model  : {model}  |  Conf threshold: {conf}")

    # --- Video info ---
    info = get_video_info(input_video)
    fps = info["fps"]
    total = info["total_frames"]
    print(f"Video  : {info['width']}x{info['height']}  {total} frames @ {fps:.1f} fps  ({info['duration_sec']:.1f}s)\n")

    # --- Init models ---
    print("Loading YOLOv8 detector...")
    detector = CarDetector(model_name=model, conf_threshold=conf)

    print("Loading DeepSORT tracker...")
    tracker = CarTracker(max_age=70, n_init=5, max_cosine_distance=0.3)

    # --- Process ---
    cap = cv2.VideoCapture(input_video)
    annotated_frames = []
    pred_frames = {}
    frame_id = 0
    limit = min(total, max_frames) if max_frames else total

    print(f"\nProcessing {limit} frames...")
    with tqdm(total=limit, unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_id >= max_frames):
                break

            detections = detector.detect(frame)
            tracks     = tracker.update(detections, frame)

            pred_frames[frame_id] = tracks
            annotated_frames.append(draw_tracks(frame, tracks))

            frame_id += 1
            pbar.update(1)

    cap.release()

    # --- Write output video ---
    write_video(annotated_frames, output_video, fps)

    # --- Basic stats ---
    print("\n=== Results ===")
    stats = count_basic_stats(pred_frames)
    print(f"  Frames processed       : {stats['total_frames']}")
    print(f"  Total detections       : {stats['total_detections']}")
    print(f"  Total unique track IDs : {stats['total_unique_tracks']}")
    print(f"  Max cars in one frame  : {stats['max_simultaneous_cars']}")
    print(f"  Avg cars per frame     : {stats['avg_simultaneous_cars']}")

    # --- Track stability metrics ---
    print("\n=== Track Stability (no GT required) ===")
    stability = compute_track_stability(pred_frames)
    print(f"  Avg track length       : {stability['avg_track_length']} frames")
    print(f"  Fragmentation ratio    : {stability['fragmentation']}  (ideal ≈ 1.0)")
    print(f"  Short tracks (<10f)    : {stability['short_track_ratio']*100:.0f}% of all tracks")
    print(f"  Detection rate         : {stability['detection_rate']*100:.0f}% of frames have detections")

    # --- GT metrics (optional) ---
    if gt_path:
        print(f"\nLoading ground truth from {gt_path} ...")
        gt_frames = load_mot_gt(gt_path)
        metrics = compute_metrics_with_gt(gt_frames, pred_frames)
        print(f"\n=== MOT Metrics ===")
        print(f"  MOTA        : {metrics['MOTA']:.2f}%")
        print(f"  IDF1        : {metrics['IDF1']:.2f}%")
        print(f"  ID Switches : {metrics['ID_Switches']}")
        print(f"  Misses      : {metrics['Misses']}")
        print(f"  False Pos.  : {metrics['FP']}")

    print(f"\nDone. Output: {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Car Detection & Tracking (CSC 528)")
    parser.add_argument("--input",      required=True,                help="Path to input F1 video")
    parser.add_argument("--output",     default="output_tracked.mp4", help="Output video path")
    parser.add_argument("--model",      default="yolov8n.pt",         help="YOLOv8 model variant")
    parser.add_argument("--conf",       type=float, default=0.25,     help="Detection confidence threshold")
    parser.add_argument("--max-frames", type=int,   default=None,     help="Limit frames (for testing)")
    parser.add_argument("--gt",         default=None,                 help="Ground truth .txt (MOT format) for MOTA/IDF1")
    args = parser.parse_args()

    run_pipeline(
        input_video=args.input,
        output_video=args.output,
        model=args.model,
        conf=args.conf,
        max_frames=args.max_frames,
        gt_path=args.gt,
    )
