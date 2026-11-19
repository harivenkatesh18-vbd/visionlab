
import cv2

from modules.image_processing import convert_to_grayscale


def detect_edges(image, lower_threshold=100, upper_threshold=200):
    """Detect edges in an image using the Canny algorithm."""

    if lower_threshold < 0 or upper_threshold > 255:
        raise ValueError("Thresholds must be between 0 and 255.")

    if lower_threshold >= upper_threshold:
        raise ValueError("Lower threshold must be less than upper threshold.")

    gray = convert_to_grayscale(image)

    edges = cv2.Canny(
        gray,
        lower_threshold,
        upper_threshold
    )

    return edges
