"""
Football Object Detection — Streamlit Web Application
======================================================
An interactive web app that uses a custom-trained YOLO11 model to detect
football-related objects in images and videos.

Detected classes:
    0: ball
    1: goalkeeper
    2: referee
    3: team 1
    4: team 2

Model: YOLO11n fine-tuned on a Roboflow football dataset (~19,800 training images).
Framework: Ultralytics >= 8.2.0 | Python >= 3.12
"""

from pathlib import Path
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = Path("best.pt")
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
SUPPORTED_VIDEO_TYPES = ["mp4"]


# ---------------------------------------------------------------------------
# Model loading (cached so it only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model() -> YOLO:
    """Load the YOLO model from disk.

    Uses Streamlit's cache_resource so the heavy model object is loaded only
    once and shared across all reruns / users.
    """
    if not MODEL_PATH.exists():
        st.error(
            f"⚠️ Model weights file **`{MODEL_PATH}`** not found! "
            "Please ensure the trained `best.pt` file is placed in the "
            "application root directory alongside `app.py`."
        )
        st.stop()
    return YOLO(str(MODEL_PATH))


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def detect_image(
    model: YOLO,
    image: np.ndarray,
    conf: float,
) -> np.ndarray:
    """Run object detection on a single image and return the annotated frame.

    Parameters
    ----------
    model : YOLO
        The loaded YOLO model instance.
    image : np.ndarray
        Input image in RGB format.
    conf : float
        Confidence threshold for predictions.

    Returns
    -------
    np.ndarray
        Annotated image with bounding boxes drawn (RGB).
    """
    results = model.predict(source=image, conf=conf, verbose=False)
    annotated = results[0].plot()  # Returns BGR numpy array
    return annotated


def process_video(
    model: YOLO,
    video_path: str,
    conf: float,
    progress_bar: st.delta_generator.DeltaGenerator,
) -> str | None:
    """Process a video frame-by-frame and write an annotated output video.

    Parameters
    ----------
    model : YOLO
        The loaded YOLO model instance.
    video_path : str
        Path to the uploaded video file on disk.
    conf : float
        Confidence threshold for predictions.
    progress_bar : st.delta_generator.DeltaGenerator
        A Streamlit progress bar element to update during processing.

    Returns
    -------
    str | None
        Path to the annotated output video, or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Failed to open the uploaded video file.")
        return None

    # Read video metadata
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a temporary output file (mp4v codec wrapped in .mp4)
    tmp_out = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp4", dir="."
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out.name, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO expects BGR input — cv2.VideoCapture already provides BGR
        results = model.predict(source=frame, conf=conf, verbose=False)
        annotated_frame = results[0].plot()
        writer.write(annotated_frame)

        # Update progress bar
        frame_idx += 1
        if total_frames > 0:
            progress_bar.progress(
                min(frame_idx / total_frames, 1.0),
                text=f"Processing frame {frame_idx}/{total_frames}",
            )

    cap.release()
    writer.release()

    # Re-encode to H.264 so that st.video() can play it in-browser.
    # mp4v codec is not browser-compatible; we transcode via OpenCV → H.264.
    h264_path = tmp_out.name.replace(".mp4", "_h264.mp4")
    _reencode_to_h264(tmp_out.name, h264_path)

    # Clean up the intermediate mp4v file
    Path(tmp_out.name).unlink(missing_ok=True)

    return h264_path


def _reencode_to_h264(input_path: str, output_path: str) -> None:
    """Re-encode a video from mp4v to H.264 using OpenCV.

    Browsers require H.264/AVC inside an MP4 container for native playback.
    If the `avc1` codec is unavailable, the function falls back to copying
    the original file so that the user still gets *some* output.
    """
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Try H.264 codec identifiers in order of availability
    for codec in ("avc1", "H264", "X264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            break
        writer.release()
    else:
        # Absolute fallback: just copy the file
        cap.release()
        import shutil
        shutil.copy2(input_path, output_path)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def build_sidebar() -> tuple[float, st.runtime.uploaded_file_manager.UploadedFile | None, str]:
    """Build the sidebar UI and return user selections.

    Returns
    -------
    tuple
        (confidence_threshold, uploaded_file, input_type)
    """
    with st.sidebar:
        st.header("⚙️ Settings")

        # Confidence threshold slider
        conf_threshold = st.slider(
            label="Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum prediction confidence to display a detection.",
        )

        st.divider()

        # Input type selector
        input_type = st.radio(
            label="Input Type",
            options=["📷 Image", "🎥 Video"],
            horizontal=True,
        )

        st.divider()

        # File uploader
        if input_type == "📷 Image":
            uploaded = st.file_uploader(
                label="Upload an image",
                type=SUPPORTED_IMAGE_TYPES,
                help="Supported formats: JPG, JPEG, PNG",
            )
        else:
            uploaded = st.file_uploader(
                label="Upload a video",
                type=SUPPORTED_VIDEO_TYPES,
                help="Supported format: MP4 (short clips recommended)",
            )

        st.divider()

        # Model info
        st.subheader("ℹ️ Model Info")
        st.markdown(
            """
            | Property | Value |
            |----------|-------|
            | **Architecture** | YOLO11n |
            | **Dataset** | Roboflow Football |
            | **Classes** | 5 |
            | **Image Size** | 640 × 640 |
            | **mAP@50** | 0.844 |
            """
        )

    return conf_threshold, uploaded, input_type


def main() -> None:
    """Main application entry point."""

    # ----- Page configuration -----
    st.set_page_config(
        page_title="⚽ Football Object Detection",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ----- Header -----
    st.title("⚽ Football Object Detection")
    st.markdown(
        "Detect and classify football-related objects in images and videos "
        "using a custom **YOLO11** model. The model identifies: "
        "**Ball**, **Goalkeeper**, **Referee**, **Team 1**, and **Team 2**."
    )
    st.divider()

    # ----- Load model -----
    model = load_model()

    # ----- Sidebar controls -----
    conf_threshold, uploaded_file, input_type = build_sidebar()

    # ----- Main content area -----
    if uploaded_file is None:
        # Show a friendly placeholder when no file is uploaded
        st.info(
            "👈 Upload an image or video from the sidebar to get started.",
            icon="📤",
        )
        return

    # -------------------- IMAGE DETECTION --------------------
    if input_type == "📷 Image":
        # Read the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # Run detection
        with st.spinner("🔍 Running detection…"):
            annotated = detect_image(model, image_np, conf_threshold)

        # Display results side-by-side
        col_orig, col_det = st.columns(2)
        with col_orig:
            st.subheader("Original Image")
            st.image(image_np, use_container_width=True)
        with col_det:
            st.subheader("Detection Results")
            st.image(annotated, channels="BGR", use_container_width=True)

        # Detection summary below the images
        results = model.predict(source=image_np, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            class_names = results[0].names
            st.success(f"✅ **{len(boxes)}** object(s) detected.")

            # Build a summary table
            detections: list[dict[str, str | float]] = []
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                detections.append({
                    "Class": class_names.get(cls_id, f"class_{cls_id}"),
                    "Confidence": f"{confidence:.2%}",
                })
            st.dataframe(detections, use_container_width=True)
        else:
            st.warning("No objects detected at the current confidence threshold.")

    # -------------------- VIDEO DETECTION --------------------
    else:
        st.subheader("🎥 Video Processing")

        # Save uploaded video to a temporary file
        tmp_video = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4", dir="."
        )
        tmp_video.write(uploaded_file.read())
        tmp_video.flush()

        # Show the original video
        st.markdown("**Original Video**")
        st.video(tmp_video.name)

        # Process the video
        st.markdown("**Processing…**")
        progress = st.progress(0.0, text="Initialising…")

        output_path = process_video(
            model=model,
            video_path=tmp_video.name,
            conf=conf_threshold,
            progress_bar=progress,
        )

        # Clean up the uploaded temp file
        Path(tmp_video.name).unlink(missing_ok=True)

        if output_path and Path(output_path).exists():
            progress.progress(1.0, text="✅ Processing complete!")
            st.markdown("**Annotated Video**")
            st.video(output_path)

            # Offer download
            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Annotated Video",
                    data=f,
                    file_name="football_detection_output.mp4",
                    mime="video/mp4",
                )

            # Clean up the output file
            Path(output_path).unlink(missing_ok=True)
        else:
            st.error(
                "❌ Video processing failed. "
                "Please try a shorter clip or a different file."
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
