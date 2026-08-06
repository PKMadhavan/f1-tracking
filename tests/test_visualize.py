import cv2
import numpy as np

from f1_tracking.visualize import draw_tracks, write_video


def test_write_video_produces_browser_playable_h264(tmp_path):
    # Regression test: write_video previously defaulted to the "mp4v" fourcc
    # (MPEG-4 Part 2), which OpenCV/ffmpeg can read back fine but which
    # Chrome/Safari refuse to decode in an HTML5 <video> tag — it shows up
    # as a player stuck at 0:00 with no visible frame. Assert the output
    # is actually H.264, which browsers do support.
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(5)]
    output_path = str(tmp_path / "out.mp4")

    write_video(frames, output_path, fps=10.0)

    cap = cv2.VideoCapture(output_path)
    assert cap.isOpened()
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert fourcc_str in ("avc1", "H264", "h264"), f"unexpected fourcc: {fourcc_str}"
    assert frame_count == len(frames)


def test_write_video_rejects_empty_frame_list(tmp_path):
    try:
        write_video([], str(tmp_path / "out.mp4"), fps=10.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_draw_tracks_does_not_mutate_input_frame():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    tracks = [{"track_id": 1, "bbox": [5, 5, 20, 20], "conf": 0.9}]

    out = draw_tracks(frame, tracks)

    assert not np.array_equal(out, frame)
    assert np.array_equal(frame, np.zeros((50, 50, 3), dtype=np.uint8))
