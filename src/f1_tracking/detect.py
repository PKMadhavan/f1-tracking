import logging

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class CarDetector:
    """
    Wraps YOLOv8 to return car detections per frame.
    COCO class 2 = 'car'. No fine-tuning needed for baseline.
    """

    CAR_CLASS_ID = 2  # COCO class index for 'car'

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.3, device: str = "cpu"):
        """
        Args:
            model_name:      YOLOv8 variant. Auto-downloads from Ultralytics on first run.
                             Options: yolov8n.pt (fast), yolov8s.pt, yolov8m.pt (accurate)
            conf_threshold:  Minimum detection confidence. 0.3 is intentionally low —
                             DeepSORT filters noise across frames. Raise to 0.5 if too
                             many false positives on barriers/ads.
            device:          'cuda' for T4 GPU on Colab, 'cpu' as fallback.
        """
        logger.info("Loading YOLOv8 detector: %s (device=%s, conf=%.2f)", model_name, device, conf_threshold)
        self.model = YOLO(model_name)
        self.conf = conf_threshold
        self.device = device

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection on a single BGR frame (as returned by cv2.VideoCapture).

        Returns:
            List of dicts, one per detected car:
            {
                "bbox": [x1, y1, x2, y2],   # absolute pixel coords
                "conf": float,               # detection confidence
                "class_id": int              # always CAR_CLASS_ID (2)
            }
        """
        if frame is None or frame.size == 0:
            raise ValueError("detect() received an empty frame")

        results = self.model.predict(
            source=frame,
            conf=self.conf,
            classes=[self.CAR_CLASS_ID],
            device=self.device,
            verbose=False,
        )
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu())
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "class_id": self.CAR_CLASS_ID,
                }
            )
        return detections
