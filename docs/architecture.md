
# VisionLab Architecture

## Overview
VisionLab is a modular computer vision application built using Python, OpenCV, and Streamlit.

## Application Architecture

- **Frontend:** Streamlit interface
- **Processing:** OpenCV and NumPy
- **AI Detection:** YOLOv8
- **Testing:** Pytest

## Main Components

| Component | Responsibility |
|---|---|
| app.py | User interface and application flow |
| image_processing.py | Image transformations |
| edge_detection.py | Canny edge detection |
| contour_detection.py | Contour extraction |
| shape_detection.py | Geometric shape recognition |
| color_detection.py | HSV-based color extraction |
| video_processing.py | Video processing and tracking |
| object_detection.py | YOLO-based object detection |
| image_utils.py | Image utility functions |

## Processing Flow

1. User uploads an image or video.
2. Streamlit receives the input.
3. The selected processing module handles it.
4. The processed result is displayed.
5. Users can download supported outputs.
