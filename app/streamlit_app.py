"""Streamlit demo app for the F1 car detection & tracking pipeline.

Run locally with:
    streamlit run app/streamlit_app.py

Assumes the f1-tracking package is installed (e.g. `pip install -e ".[app]"`)
so that `f1_tracking` imports cleanly without any sys.path manipulation.
"""

import tempfile
from pathlib import Path

import streamlit as st

from f1_tracking.pipeline import run_pipeline
from f1_tracking.utils import get_video_info

# Rough observed YOLOv8n throughput on CPU, used only to show the user a
# time estimate before they commit to a long run — not a hard guarantee.
EST_CPU_FPS = 12

st.set_page_config(
    page_title="F1 Car Detection & Tracking",
    page_icon="🏎️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🏎️ F1 Car Detection & Tracking")
st.markdown(
    "Upload a clip of Formula 1 race footage and this app will detect and track "
    "the cars frame-by-frame, drawing bounding boxes and persistent track IDs on "
    "the output video."
)

with st.expander("About this demo", expanded=False):
    st.markdown("""
This demo runs a **YOLOv8 + DeepSORT** pipeline:

- **YOLOv8** (Ultralytics) performs per-frame object detection to find car bounding boxes.
- **DeepSORT** associates those detections across frames into consistent tracks, assigning
  each car a stable ID even through brief occlusions or camera cuts.
- After processing, the app reports basic detection statistics and track-stability
  metrics (average track length, fragmentation, detection rate) computed directly
  from the tracker output — no ground-truth annotations required.

Everything runs on CPU by default, so by default the app tracks the entire clip you
upload — a 30-second clip can take a couple of minutes at this throughput. Use the
"Process entire video" toggle to cap the run to fewer frames for a quicker test.
        """)

# --------------------------------------------------------------------------
# Upload (first, so the video's real length is known before showing controls)
# --------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload race video",
    type=["mp4", "mov", "avi"],
    help="MP4, MOV, or AVI. The whole clip is tracked by default, however long it is.",
)

if uploaded_file is not None:
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("input_file_key") != file_key:
        input_suffix = Path(uploaded_file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix) as tmp_in:
            tmp_in.write(uploaded_file.getbuffer())
            st.session_state["input_path"] = tmp_in.name
        st.session_state["input_file_key"] = file_key
        try:
            st.session_state["video_info"] = get_video_info(st.session_state["input_path"])
        except Exception:  # noqa: BLE001 - best-effort metadata read, shouldn't block upload
            st.session_state["video_info"] = None

    st.caption(f"Loaded: {uploaded_file.name} ({uploaded_file.size / 1e6:.1f} MB)")

video_info = st.session_state.get("video_info") if uploaded_file is not None else None
total_frames_available = video_info["total_frames"] if video_info else None

if video_info:
    st.caption(
        f"Video: {video_info['width']}x{video_info['height']}, {video_info['total_frames']} frames "
        f"@ {video_info['fps']:.1f} fps ({video_info['duration_sec']:.1f}s)"
    )

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.header("Settings")

model = st.sidebar.selectbox(
    "Model",
    options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
    index=0,
    help=(
        "YOLOv8 variant to use for detection. `yolov8n` (nano) is fastest and the "
        "recommended default for a live CPU demo. `yolov8s`/`yolov8m` are more "
        "accurate but noticeably slower per frame on CPU."
    ),
)

conf = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.9,
    value=0.25,
    step=0.01,
    help=(
        "Minimum detection confidence to keep a box. Lower values catch more cars "
        "(better recall) but add false positives; higher values are stricter and "
        "may miss fast-moving or partially occluded cars."
    ),
)

process_full = st.sidebar.checkbox(
    "Process entire video",
    value=True,
    help="Track every frame of the uploaded video, however long it is.",
)

max_frames = None
if not process_full:
    upper_bound = total_frames_available if total_frames_available else 5000
    default_cap = min(150, upper_bound)
    max_frames = st.sidebar.number_input(
        "Max frames to process",
        min_value=10,
        max_value=max(upper_bound, 10),
        value=default_cap,
        step=10,
        help="Caps how many frames of the uploaded video are processed, for a quicker test run.",
    )

st.sidebar.markdown("---")
if total_frames_available:
    frames_to_run = int(max_frames) if max_frames is not None else total_frames_available
    est_seconds = frames_to_run / EST_CPU_FPS
    st.sidebar.caption(
        f"Estimated time: ~{est_seconds / 60:.1f} min for {frames_to_run} frames "
        f"at ~{EST_CPU_FPS} fps on CPU with `yolov8n.pt`."
    )
else:
    st.sidebar.caption("Upload a video to see a time estimate.")

run_clicked = st.button("Run Tracking", type="primary", disabled=uploaded_file is None)

# --------------------------------------------------------------------------
# Run pipeline
# --------------------------------------------------------------------------
if run_clicked and uploaded_file is not None:
    input_path = st.session_state["input_path"]
    output_path = str(Path(tempfile.gettempdir()) / f"f1_tracked_{Path(input_path).stem}.mp4")

    progress_bar = st.progress(0, text="Starting...")
    status_text = st.empty()

    def _on_progress(frame_id: int, total_frames: int) -> None:
        pct = min(frame_id / total_frames, 1.0) if total_frames else 0.0
        progress_bar.progress(pct, text=f"Processing frame {frame_id}/{total_frames}")

    try:
        with st.spinner("Running detection & tracking... this may take a while on CPU."):
            result = run_pipeline(
                input_video=input_path,
                output_video=output_path,
                model=model,
                conf=conf,
                max_frames=int(max_frames) if max_frames is not None else None,
                gt_path=None,
                progress_callback=_on_progress,
            )
        progress_bar.progress(1.0, text="Done!")
        status_text.success("Tracking complete.")

        st.subheader("Annotated Output")
        st.video(result["output_video"])

        stats = result["stats"]
        stability = result["stability"]

        st.subheader("Detection Statistics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Frames processed", stats["total_frames"])
        c2.metric("Total detections", stats["total_detections"])
        c3.metric("Unique tracks", stats["total_unique_tracks"])
        c4.metric("Max cars/frame", stats["max_simultaneous_cars"])
        c5.metric("Avg cars/frame", f"{stats['avg_simultaneous_cars']:.2f}")

        st.subheader("Track Stability")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Avg track length", f"{stability['avg_track_length']:.1f} frames")
        s2.metric("Fragmentation", f"{stability['fragmentation']:.2f}", help="Ideal ~= 1.0")
        s3.metric(
            "Short-track ratio", f"{stability['short_track_ratio'] * 100:.0f}%", help="Tracks under 10 frames"
        )
        s4.metric(
            "Detection rate",
            f"{stability['detection_rate'] * 100:.0f}%",
            help="Frames with at least one detection",
        )

        if result.get("gt_metrics"):
            gt = result["gt_metrics"]
            st.subheader("MOT Metrics (Ground Truth)")
            g1, g2, g3, g4, g5 = st.columns(5)
            g1.metric("MOTA", f"{gt['MOTA']:.2f}%")
            g2.metric("IDF1", f"{gt['IDF1']:.2f}%")
            g3.metric("ID Switches", gt["ID_Switches"])
            g4.metric("Misses", gt["Misses"])
            g5.metric("False Positives", gt["FP"])

    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the UI
        progress_bar.empty()
        st.error(f"Pipeline failed: {exc}")
