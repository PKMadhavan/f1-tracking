"""
Unit tests for CarDetector / CarTracker with ultralytics.YOLO and DeepSort mocked
out, so tests run fast in CI with no model download and no GPU required.
"""

import numpy as np
import pytest


class _FakeTensor:
    """Minimal stand-in for a torch.Tensor supporting the .cpu()/.numpy()/float() chain
    that detect.py and ground_truth.py call on YOLO box outputs."""

    def __init__(self, value):
        self._value = value

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self._value)

    def __float__(self):
        return float(self._value)


def _make_fake_box(bbox, conf):
    box = type("FakeBox", (), {})()
    box.xyxy = [_FakeTensor(bbox)]
    box.conf = [_FakeTensor(conf)]
    return box


@pytest.fixture
def mock_yolo(mocker):
    mock_cls = mocker.patch("f1_tracking.detect.YOLO")
    mock_instance = mock_cls.return_value
    return mock_cls, mock_instance


@pytest.fixture
def mock_deepsort(mocker):
    mock_cls = mocker.patch("f1_tracking.track.DeepSort")
    mock_instance = mock_cls.return_value
    return mock_cls, mock_instance


def test_car_detector_returns_parsed_detections(mock_yolo):
    from f1_tracking.detect import CarDetector

    mock_cls, mock_instance = mock_yolo
    fake_result = type("FakeResult", (), {})()
    fake_result.boxes = [_make_fake_box([10, 20, 110, 220], 0.87)]
    mock_instance.predict.return_value = [fake_result]

    detector = CarDetector(model_name="yolov8n.pt", conf_threshold=0.3)
    mock_cls.assert_called_once_with("yolov8n.pt")

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0]["bbox"] == [10, 20, 110, 220]
    assert detections[0]["conf"] == pytest.approx(0.87)
    assert detections[0]["class_id"] == CarDetector.CAR_CLASS_ID


def test_car_detector_rejects_empty_frame(mock_yolo):
    from f1_tracking.detect import CarDetector

    detector = CarDetector()
    with pytest.raises(ValueError):
        detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))


def test_car_detector_no_detections(mock_yolo):
    from f1_tracking.detect import CarDetector

    _, mock_instance = mock_yolo
    fake_result = type("FakeResult", (), {})()
    fake_result.boxes = []
    mock_instance.predict.return_value = [fake_result]

    detector = CarDetector()
    detections = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert detections == []


def _make_fake_track(track_id, bbox, confirmed=True, det_conf=0.9):
    track = type("FakeTrack", (), {})()
    track.track_id = track_id
    track.det_conf = det_conf
    track.is_confirmed = lambda: confirmed
    track.to_ltrb = lambda: [float(v) for v in bbox]
    return track


def test_car_tracker_returns_confirmed_tracks(mock_deepsort):
    from f1_tracking.track import CarTracker

    _mock_cls, mock_instance = mock_deepsort
    mock_instance.update_tracks.return_value = [
        _make_fake_track("1", [10, 20, 110, 220]),
        _make_fake_track("2", [0, 0, 10, 10], confirmed=False),  # should be filtered out
    ]

    tracker = CarTracker(max_age=70, n_init=5, max_cosine_distance=0.3)
    detections = [{"bbox": [10, 20, 110, 220], "conf": 0.8, "class_id": 2}]
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    results = tracker.update(detections, frame)

    assert len(results) == 1
    assert results[0]["track_id"] == "1"
    assert results[0]["bbox"] == [10, 20, 110, 220]
    assert results[0]["conf"] == pytest.approx(0.9)


def test_car_tracker_converts_bbox_to_xywh_for_deepsort(mock_deepsort):
    from f1_tracking.track import CarTracker

    _, mock_instance = mock_deepsort
    mock_instance.update_tracks.return_value = []

    tracker = CarTracker()
    detections = [{"bbox": [10, 20, 110, 220], "conf": 0.8, "class_id": 2}]
    tracker.update(detections, np.zeros((240, 320, 3), dtype=np.uint8))

    raw_arg, _kwargs = mock_instance.update_tracks.call_args
    raw = raw_arg[0]
    assert raw == [([10, 20, 100, 200], 0.8, 2)]
