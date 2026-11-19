
# VisionLab Implementation

## Technologies
- Python
- OpenCV
- NumPy
- Streamlit
- Ultralytics YOLOv8
- Pytest

## Implemented Features

### Image Processing
- Grayscale conversion
- Gaussian and median blur
- Image sharpening
- Brightness and contrast adjustment
- Thresholding
- Image resizing

### Computer Vision
- Canny edge detection
- Contour detection
- Shape classification
- HSV color detection

### Video Processing
- Grayscale video conversion
- Edge detection
- Gaussian blur
- Red-object tracking

### AI-Based Detection
- YOLOv8 object detection
- Adjustable confidence threshold
- Bounding-box visualization
- Detection labels and confidence scores

## Testing
The project uses Pytest for automated testing of processing modules and video functionality.

## Current Limitations
- Object detection accuracy depends on image quality and pretrained model classes.
- Red-object tracking is designed for red-colored objects.
- Shape detection works best with clear geometric shapes.

## Future Improvements
- Custom-trained object detection models
- More robust object tracking
- Additional image enhancement techniques
- Expanded automated test coverage
