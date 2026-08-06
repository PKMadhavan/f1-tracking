from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np


class CarTracker:
    """
    Wraps DeepSORT for multi-object tracking across video frames.
    Kalman filter predicts positions; Hungarian algorithm matches detections;
    CNN appearance embedding handles re-ID after occlusion or camera cuts.
    """

    def __init__(
        self,
        max_age=70,
        n_init=5,
        max_cosine_distance=0.3,
        embedder="mobilenet",
        half=False,
    ):
        """
        Args:
            max_age:              Frames to keep a track alive without a detection match.
                                  70 frames (~1.4s at 50fps) covers typical F1 camera cuts.
            n_init:               Consecutive detections required before confirming a track.
                                  5 reduces ghost tracks from motion-blur false positives.
            max_cosine_distance:  Appearance similarity threshold for re-ID.
                                  0.3 is stricter — prevents ID swaps on near-identical liveries.
            embedder:             CNN for appearance features.
                                  'mobilenet' = fast; 'clip_RN50' = more accurate but slower.
            half:                 FP16 inference — only useful on GPU, set False for CPU.
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            embedder=embedder,
            half=half,
            embedder_gpu=False,
        )

    def update(self, detections: list, frame: np.ndarray) -> list:
        """
        Update tracker with new detections from one frame.

        Args:
            detections: Output of CarDetector.detect() — list of bbox/conf/class dicts.
            frame:      Current BGR frame (needed for CNN appearance embedding).

        Returns:
            List of confirmed active tracks:
            {
                "track_id": int,
                "bbox": [x1, y1, x2, y2],
                "conf": float
            }
        """
        # DeepSORT expects: list of ([x1, y1, w, h], conf, class_id)
        raw = [
            (
                [
                    d["bbox"][0],
                    d["bbox"][1],
                    d["bbox"][2] - d["bbox"][0],   # width
                    d["bbox"][3] - d["bbox"][1],   # height
                ],
                d["conf"],
                d["class_id"]
            )
            for d in detections
        ]

        tracks = self.tracker.update_tracks(raw, frame=frame)

        results = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            results.append({
                "track_id": t.track_id,
                "bbox": [x1, y1, x2, y2],
                "conf": t.det_conf if t.det_conf else 0.0
            })
        return results
