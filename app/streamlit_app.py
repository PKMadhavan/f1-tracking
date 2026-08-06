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

Everything runs on CPU by default, so keep clips short and frame counts modest for a
snappy live demo. For full-video, GPU-accelerated runs, use the `f1-track` CLI that
ships with this package instead.
        """)

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

max_frames = st.sidebar.number_input(
    "Max frames to process",
    min_value=10,
    max_value=5000,
    value=150,
    step=10,
    help=(
        "Caps how many frames of the uploaded video are processed. This app runs "
        "on CPU, where YOLOv8 typically manages only a few frames per second, so "
        "keep this small (e.g. 150) for a responsive live demo. Increase it only "
        "if you're prepared to wait — full videos can take many minutes on CPU."
    ),
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: `yolov8n.pt` at ~150 frames typically finishes in well under a minute on CPU.")

# --------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload race video",
    type=["mp4", "mov", "avi"],
    help="MP4, MOV, or AVI. Shorter clips process faster on CPU.",
)

run_clicked = st.button("Run Tracking", type="primary", disabled=uploaded_file is None)

if uploaded_file is not None:
    st.caption(f"Loaded: {uploaded_file.name} ({uploaded_file.size / 1e6:.1f} MB)")

# --------------------------------------------------------------------------
# Run pipeline
# --------------------------------------------------------------------------
if run_clicked and uploaded_file is not None:
    input_suffix = Path(uploaded_file.name).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix) as tmp_in:
        tmp_in.write(uploaded_file.getbuffer())
        input_path = tmp_in.name

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
                max_frames=int(max_frames),
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
