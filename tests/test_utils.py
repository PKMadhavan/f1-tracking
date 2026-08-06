import cv2
import numpy as np
import pytest

from f1_tracking.utils import ensure_dir, get_video_info, xywh_to_xyxy, xyxy_to_xywh


def test_xyxy_to_xywh():
    assert xyxy_to_xywh([10, 20, 110, 220]) == [10, 20, 100, 200]


def test_xywh_to_xyxy():
    assert xywh_to_xyxy([10, 20, 100, 200]) == [10, 20, 110, 220]


def test_bbox_roundtrip():
    original = [5, 15, 205, 315]
    assert xywh_to_xyxy(xyxy_to_xywh(original)) == original


def test_ensure_dir_creates_directory(tmp_path):
    target = tmp_path / "nested" / "dir"
    assert not target.exists()
    ensure_dir(str(target))
    assert target.is_dir()


def test_ensure_dir_is_idempotent(tmp_path):
    target = tmp_path / "dir"
    ensure_dir(str(target))
    ensure_dir(str(target))  # should not raise
    assert target.is_dir()


def test_get_video_info_missing_file():
    with pytest.raises(FileNotFoundError):
        get_video_info("/nonexistent/path/does_not_exist.mp4")


def test_get_video_info_reads_metadata(tmp_path):
    # MJPG/.avi is the most broadly supported OpenCV codec+container combo
    # across platforms/builds that lack a system ffmpeg — mp4v isn't reliably
    # available in opencv-python-headless wheels.
    video_path = str(tmp_path / "synthetic.avi")
    fps, width, height, n_frames = 10.0, 64, 48, 5

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    assert writer.isOpened(), "test environment cannot write video via OpenCV"
    for _ in range(n_frames):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()

    info = get_video_info(video_path)
    assert info["width"] == width
    assert info["height"] == height
    assert info["total_frames"] == n_frames
    assert info["fps"] == pytest.approx(fps, rel=0.1)
