"""
compare_models.py — Benchmark yolov8n vs yolov8s on F1 footage.

Measures detections per frame, confidence distributions, and inference speed.
Outputs comparison table + saves comparison_plot.png.

Usage:
    python compare_models.py --input f1_trimmed.mp4 --frames 300
"""

import cv2
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm


CAR_CLASS_ID = 2
MODELS = ["yolov8n.pt", "yolov8s.pt"]


def benchmark(video_path, model_name, conf, n_frames):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video_path)

    det_counts = []
    conf_scores = []
    times = []

    for _ in tqdm(range(n_frames), desc=model_name, leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        results = model.predict(source=frame, conf=conf, classes=[CAR_CLASS_ID],
                                device="cpu", verbose=False)
        t1 = time.perf_counter()
        boxes = results[0].boxes
        det_counts.append(len(boxes))
        for box in boxes:
            conf_scores.append(float(box.conf[0].cpu()))
        times.append(t1 - t0)

    cap.release()
    return {
        "model":         model_name,
        "avg_dets":      round(np.mean(det_counts), 2),
        "max_dets":      int(np.max(det_counts)),
        "avg_conf":      round(np.mean(conf_scores), 3) if conf_scores else 0,
        "avg_ms":        round(np.mean(times) * 1000, 1),
        "fps":           round(1 / np.mean(times), 1),
        "det_counts":    det_counts,
        "conf_scores":   conf_scores,
    }


def run_comparison(video_path, conf=0.25, n_frames=300):
    print(f"\n=== Model Comparison: yolov8n vs yolov8s ===")
    print(f"Video : {video_path}  |  Frames: {n_frames}  |  Conf: {conf}\n")

    results = [benchmark(video_path, m, conf, n_frames) for m in MODELS]

    # Print table
    print(f"\n{'Model':<14} {'Avg dets/frame':>14} {'Max dets':>9} {'Avg conf':>9} {'Avg ms':>8} {'FPS':>6}")
    print("-" * 65)
    for r in results:
        print(f"{r['model']:<14} {r['avg_dets']:>14} {r['max_dets']:>9} "
              f"{r['avg_conf']:>9} {r['avg_ms']:>8} {r['fps']:>6}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["royalblue", "darkorange"]

    # Detections per frame
    for r, c in zip(results, colors):
        axes[0].plot(r["det_counts"], color=c, alpha=0.8, label=r["model"], linewidth=1)
    axes[0].set_title("Detections Per Frame")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Cars detected")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Confidence distribution
    for r, c in zip(results, colors):
        if r["conf_scores"]:
            axes[1].hist(r["conf_scores"], bins=25, color=c, alpha=0.6,
                         label=r["model"], edgecolor="white")
    axes[1].set_title("Detection Confidence Distribution")
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Speed comparison bar chart
    labels = [r["model"] for r in results]
    fps_vals = [r["fps"] for r in results]
    bars = axes[2].bar(labels, fps_vals, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, fps_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f"{val} fps", ha="center", fontsize=11, fontweight="bold")
    axes[2].set_title("Inference Speed (CPU)")
    axes[2].set_ylabel("Frames per second")
    axes[2].set_ylim(0, max(fps_vals) * 1.3)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.suptitle("YOLOv8n vs YOLOv8s — F1 Detection Benchmark", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("comparison_plot.png", dpi=150)
    plt.close()
    print("\nSaved → comparison_plot.png")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,      help="Input video")
    parser.add_argument("--conf",   type=float, default=0.25)
    parser.add_argument("--frames", type=int,   default=300, help="Frames to benchmark")
    args = parser.parse_args()
    run_comparison(args.input, args.conf, args.frames)
