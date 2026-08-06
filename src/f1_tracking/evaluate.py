import motmetrics as mm
import numpy as np


def compute_metrics_with_gt(gt_frames: dict, pred_frames: dict) -> dict:
    """
    Compute MOTA, IDF1, and ID switches using ground-truth annotations.
    Requires manual annotation (e.g. from CVAT or Roboflow in MOT format).

    Args:
        gt_frames:   {frame_id: [{"bbox":[x1,y1,x2,y2], "gt_id": int}, ...]}
        pred_frames: {frame_id: [{"bbox":[x1,y1,x2,y2], "track_id": int}, ...]}

    Returns:
        Dict with: MOTA (%), IDF1 (%), ID_Switches, Misses, FP
    """
    acc = mm.MOTAccumulator(auto_id=True)

    all_frames = sorted(set(list(gt_frames.keys()) + list(pred_frames.keys())))
    for frame_id in all_frames:
        gts = gt_frames.get(frame_id, [])
        preds = pred_frames.get(frame_id, [])

        gt_ids = [g["gt_id"] for g in gts]
        pred_ids = [p["track_id"] for p in preds]

        if gts and preds:
            dist = mm.distances.iou_matrix(
                [g["bbox"] for g in gts],
                [p["bbox"] for p in preds],
                max_iou=0.5,
            )
        else:
            dist = np.empty((len(gt_ids), len(pred_ids)))

        acc.update(gt_ids, pred_ids, dist)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["mota", "idf1", "num_switches", "num_misses", "num_false_positives"],
    )

    return {
        "MOTA": round(float(summary["mota"].iloc[0]) * 100, 2),
        "IDF1": round(float(summary["idf1"].iloc[0]) * 100, 2),
        "ID_Switches": int(summary["num_switches"].iloc[0]),
        "Misses": int(summary["num_misses"].iloc[0]),
        "FP": int(summary["num_false_positives"].iloc[0]),
    }


def load_mot_gt(gt_txt_path: str) -> dict:
    """
    Load a MOT-format ground truth file into the gt_frames dict.

    MOT format: frame, id, x, y, w, h, conf, -1, -1, -1  (comma-separated)
    x, y, w, h are in pixel coords (top-left corner + width/height).

    Returns:
        {frame_id: [{"bbox":[x1,y1,x2,y2], "gt_id": int}, ...]}
    """
    gt_frames: dict = {}
    with open(gt_txt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            gt_id = int(parts[1])
            x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            bbox = [x, y, x + w, y + h]
            gt_frames.setdefault(frame_id, []).append({"bbox": bbox, "gt_id": gt_id})
    return gt_frames


def count_basic_stats(pred_frames: dict) -> dict:
    """
    Compute basic statistics without ground truth.

    Returns:
        total_unique_tracks, max_simultaneous_cars, avg_simultaneous_cars,
        total_frames, total_detections
    """
    all_ids = set()
    max_simultaneous = 0
    total_detections = 0

    for frame_id in sorted(pred_frames):
        tracks = pred_frames[frame_id]
        ids = {t["track_id"] for t in tracks}
        all_ids.update(ids)
        max_simultaneous = max(max_simultaneous, len(tracks))
        total_detections += len(tracks)

    total_frames = len(pred_frames)
    avg_simultaneous = round(total_detections / total_frames, 2) if total_frames else 0

    return {
        "total_unique_tracks": len(all_ids),
        "max_simultaneous_cars": max_simultaneous,
        "avg_simultaneous_cars": avg_simultaneous,
        "total_frames": total_frames,
        "total_detections": total_detections,
    }


def compute_track_stability(pred_frames: dict) -> dict:
    """
    Measure tracking quality without ground truth by analysing track lifespans.

    Metrics:
        avg_track_length  — average number of frames a track ID is active.
                            Higher = more stable, fewer ID switches.
        fragmentation     — ratio of unique IDs to max simultaneous cars.
                            Ideal ≈ 1.0; >3 indicates heavy ID fragmentation.
        short_track_ratio — fraction of tracks seen in <10 frames (ghost tracks).
        detection_rate    — fraction of frames that contain at least one detection.
    """
    # Build per-track frame lists
    track_frames: dict = {}
    frames_with_detections = 0

    for frame_id in sorted(pred_frames):
        tracks = pred_frames[frame_id]
        if tracks:
            frames_with_detections += 1
        for t in tracks:
            tid = t["track_id"]
            track_frames.setdefault(tid, []).append(frame_id)

    if not track_frames:
        return {
            "avg_track_length": 0,
            "fragmentation": 0,
            "short_track_ratio": 0,
            "detection_rate": 0,
        }

    lengths = [len(v) for v in track_frames.values()]
    avg_length = round(sum(lengths) / len(lengths), 1)

    max_simultaneous = max(len(pred_frames[f]) for f in pred_frames) if pred_frames else 1

    fragmentation = round(len(track_frames) / max(max_simultaneous, 1), 2)

    short_tracks = sum(1 for l in lengths if l < 10)
    short_track_ratio = round(short_tracks / len(lengths), 2)

    detection_rate = round(frames_with_detections / max(len(pred_frames), 1), 2)

    return {
        "avg_track_length": avg_length,
        "fragmentation": fragmentation,
        "short_track_ratio": short_track_ratio,
        "detection_rate": detection_rate,
    }
