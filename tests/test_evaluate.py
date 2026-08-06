from f1_tracking.evaluate import (
    compute_metrics_with_gt,
    compute_track_stability,
    count_basic_stats,
    load_mot_gt,
)


def test_count_basic_stats_empty():
    stats = count_basic_stats({})
    assert stats["total_frames"] == 0
    assert stats["total_detections"] == 0
    assert stats["total_unique_tracks"] == 0
    assert stats["avg_simultaneous_cars"] == 0


def test_count_basic_stats_typical():
    pred_frames = {
        0: [{"track_id": 1, "bbox": [0, 0, 10, 10], "conf": 0.9}],
        1: [
            {"track_id": 1, "bbox": [1, 0, 11, 10], "conf": 0.9},
            {"track_id": 2, "bbox": [50, 50, 60, 60], "conf": 0.8},
        ],
        2: [],
    }
    stats = count_basic_stats(pred_frames)
    assert stats["total_frames"] == 3
    assert stats["total_detections"] == 3
    assert stats["total_unique_tracks"] == 2
    assert stats["max_simultaneous_cars"] == 2
    assert stats["avg_simultaneous_cars"] == 1.0


def test_compute_track_stability_empty():
    stability = compute_track_stability({})
    assert stability == {
        "avg_track_length": 0,
        "fragmentation": 0,
        "short_track_ratio": 0,
        "detection_rate": 0,
    }


def test_compute_track_stability_perfect_single_track():
    # One track present in every one of 20 frames -> ideal fragmentation (1.0),
    # no short tracks, 100% detection rate.
    pred_frames = {i: [{"track_id": 1, "bbox": [0, 0, 10, 10], "conf": 0.9}] for i in range(20)}
    stability = compute_track_stability(pred_frames)
    assert stability["avg_track_length"] == 20
    assert stability["fragmentation"] == 1.0
    assert stability["short_track_ratio"] == 0
    assert stability["detection_rate"] == 1.0


def test_compute_track_stability_short_tracks_flagged():
    # A track that only appears in 3 frames (<10) should count toward short_track_ratio.
    pred_frames = {i: [{"track_id": 1, "bbox": [0, 0, 10, 10], "conf": 0.9}] for i in range(3)}
    stability = compute_track_stability(pred_frames)
    assert stability["short_track_ratio"] == 1.0


def test_load_mot_gt(tmp_path):
    gt_path = tmp_path / "gt.txt"
    gt_path.write_text("1,1,10,20,30,40,1,-1,-1,-1\n2,1,12,22,30,40,1,-1,-1,-1\n")

    gt_frames = load_mot_gt(str(gt_path))
    assert set(gt_frames.keys()) == {1, 2}
    assert gt_frames[1] == [{"bbox": [10, 20, 40, 60], "gt_id": 1}]


def test_load_mot_gt_skips_malformed_lines(tmp_path):
    gt_path = tmp_path / "gt.txt"
    gt_path.write_text("not,enough,fields\n1,1,10,20,30,40,1,-1,-1,-1\n")

    gt_frames = load_mot_gt(str(gt_path))
    assert list(gt_frames.keys()) == [1]


def test_compute_metrics_with_gt_perfect_match():
    # Prediction exactly matches ground truth in every frame -> zero ID switches,
    # zero misses, zero false positives.
    frames = {
        0: [{"bbox": [0, 0, 10, 10]}],
        1: [{"bbox": [1, 0, 11, 10]}],
    }
    gt_frames = {f: [{**item, "gt_id": 1} for item in items] for f, items in frames.items()}
    pred_frames = {f: [{**item, "track_id": 1} for item in items] for f, items in frames.items()}

    metrics = compute_metrics_with_gt(gt_frames, pred_frames)
    assert metrics["ID_Switches"] == 0
    assert metrics["Misses"] == 0
    assert metrics["FP"] == 0
    assert metrics["MOTA"] == 100.0
    assert metrics["IDF1"] == 100.0


def test_compute_metrics_with_gt_id_switch_detected():
    # Same spatial track across 2 frames, but the predicted ID changes -> 1 switch.
    gt_frames = {
        0: [{"bbox": [0, 0, 10, 10], "gt_id": 1}],
        1: [{"bbox": [1, 0, 11, 10], "gt_id": 1}],
    }
    pred_frames = {
        0: [{"bbox": [0, 0, 10, 10], "track_id": 1}],
        1: [{"bbox": [1, 0, 11, 10], "track_id": 2}],  # switched
    }
    metrics = compute_metrics_with_gt(gt_frames, pred_frames)
    assert metrics["ID_Switches"] == 1
