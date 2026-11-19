
import os
import tempfile

import cv2
import numpy as np
import streamlit as st

from modules.image_processing import (
    convert_to_grayscale,
    apply_gaussian_blur,
    apply_median_blur,
    sharpen_image,
    adjust_brightness_contrast,
    apply_threshold,
    resize_image,
)

from modules.edge_detection import detect_edges

from modules.contour_detection import (
    detect_contours,
    draw_contours,
)

from modules.shape_detection import detect_shapes

from modules.color_detection import (
    create_color_mask,
    extract_color,
)

from modules.video_processing import process_video

from modules.object_detection import detect_objects


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="VisionLab",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 VisionLab")
st.caption(
    "An AI-powered Computer Vision Playground "
    "for Image Processing, Detection and Video Analysis."
)


# ==================================================
# IMAGE PROCESSING
# ==================================================

st.header("🖼️ Image Processing")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    key="image_uploader",
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        st.error("Unable to read the uploaded image.")

    else:

        st.sidebar.header("Image Operations")

        operation = st.sidebar.selectbox(
            "Choose an operation",
            [
                "Original",
                "Grayscale",
                "Gaussian Blur",
                "Median Blur",
                "Sharpen",
                "Brightness & Contrast",
                "Edge Detection",
                "Thresholding",
                "Resize",
                "Contour Detection",
                "Shape Detection",
                "Color Detection",
                "AI Object Detection",
            ],
        )

        processed = image.copy()

        # ------------------------------------------
        # IMAGE OPERATIONS
        # ------------------------------------------

        if operation == "Original":
            processed = image.copy()

        elif operation == "Grayscale":
            processed = convert_to_grayscale(image)

        elif operation == "Gaussian Blur":
            processed = apply_gaussian_blur(image)

        elif operation == "Median Blur":
            processed = apply_median_blur(image)

        elif operation == "Sharpen":
            processed = sharpen_image(image)

        elif operation == "Brightness & Contrast":
            processed = adjust_brightness_contrast(
                image,
                brightness=30,
                contrast=1.5,
            )

        elif operation == "Edge Detection":
            processed = detect_edges(image)

        elif operation == "Thresholding":
            processed = apply_threshold(image)

        elif operation == "Resize":
            processed = resize_image(image)

        elif operation == "Contour Detection":

            contours = detect_contours(image)

            processed = draw_contours(
                image,
                contours,
            )

            st.sidebar.write(
                f"Contours detected: {len(contours)}"
            )

        elif operation == "Shape Detection":

            processed, detected_shapes = detect_shapes(
                image
            )

            st.sidebar.write("Detected Shapes:")

            if detected_shapes:
                for shape in detected_shapes:
                    st.sidebar.write(f"• {shape}")
            else:
                st.sidebar.write("No shapes detected.")

        elif operation == "Color Detection":

            lower_red = np.array(
                [0, 100, 100],
                dtype=np.uint8,
            )

            upper_red = np.array(
                [10, 255, 255],
                dtype=np.uint8,
            )

            mask = create_color_mask(
                image,
                lower_red,
                upper_red,
            )

            processed = extract_color(
                image,
                mask,
            )

        elif operation == "AI Object Detection":

            confidence = st.sidebar.slider(
                "Detection Confidence",
                min_value=0.10,
                max_value=0.90,
                value=0.25,
                step=0.05,
            )

            with st.spinner("Detecting objects with YOLO..."):

                processed, detections = detect_objects(
                    image,
                    confidence=confidence,
                )

            st.sidebar.write("Detected Objects:")

            if detections:
                for item in detections:
                    st.sidebar.write(
                        f"{item['name']} — "
                        f"{item['confidence']:.0%}"
                    )
            else:
                st.sidebar.write("No objects detected.")

        # ------------------------------------------
        # IMAGE COMPARISON
        # ------------------------------------------

        st.subheader("Image Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB,
                ),
                caption="Original Image",
                use_container_width=True,
            )

        with col2:

            if len(processed.shape) == 2:
                display_image = processed
            else:
                display_image = cv2.cvtColor(
                    processed,
                    cv2.COLOR_BGR2RGB,
                )

            st.image(
                display_image,
                caption=operation,
                use_container_width=True,
            )

        # ------------------------------------------
        # IMAGE DOWNLOAD
        # ------------------------------------------

        success, encoded_image = cv2.imencode(
            ".png",
            processed,
        )

        if success:
            st.download_button(
                label="Download Processed Image",
                data=encoded_image.tobytes(),
                file_name="visionlab_processed.png",
                mime="image/png",
            )

else:
    st.info(
        "Upload an image to explore image-processing features."
    )


# ==================================================
# VIDEO PROCESSING
# ==================================================

st.divider()

st.header("🎬 Video Processing")

st.write(
    "Upload a video and apply computer vision "
    "operations frame by frame."
)

uploaded_video = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi"],
    key="video_uploader",
)

if uploaded_video is not None:

    video_operation = st.selectbox(
        "Choose a video operation",
        [
            "grayscale",
            "edges",
            "blur",
            "tracking",
        ],
        key="video_operation",
    )

    if video_operation == "tracking":
        st.caption(
            "Red object tracking works best with a clearly "
            "visible red object against a contrasting background."
        )

    if st.button(
        "Process Video",
        key="process_video_button",
    ):

        with st.spinner("Processing video..."):

            try:

                with tempfile.TemporaryDirectory() as temp_dir:

                    input_path = os.path.join(
                        temp_dir,
                        "input_video.mp4",
                    )

                    output_path = os.path.join(
                        temp_dir,
                        "processed_video.mp4",
                    )

                    # Save uploaded video
                    with open(input_path, "wb") as file:
                        file.write(
                            uploaded_video.getvalue()
                        )

                    # Process video
                    process_video(
                        input_path,
                        output_path,
                        operation=video_operation,
                    )

                    # Read processed video
                    with open(output_path, "rb") as file:
                        processed_video = file.read()

                st.success(
                    "Video processed successfully!"
                )

                st.subheader("Processed Video")

                st.video(processed_video)

                st.download_button(
                    label="Download Processed Video",
                    data=processed_video,
                    file_name="visionlab_processed.mp4",
                    mime="video/mp4",
                )

            except Exception as error:

                st.error(
                    f"Video processing failed: {error}"
                )

else:
    st.info(
        "Upload a video to explore video-processing features."
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "VisionLab | Built with Python, OpenCV, NumPy, "
    "Streamlit and YOLO"
)
