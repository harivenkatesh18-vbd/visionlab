
import cv2

from modules.image_processing import convert_to_grayscale


def detect_contours(
    image,
    threshold1=50,
    threshold2=150,
    min_area=150
):
    """Detect significant object boundaries."""

    gray = convert_to_grayscale(image)

    # Reduce small image details and noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blurred, threshold1, threshold2)

    # Find contours
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Ignore very small contours
    filtered_contours = [
        contour for contour in contours
        if cv2.contourArea(contour) >= min_area
    ]

    return filtered_contours


def draw_contours(image, contours):
    """Draw detected contours on a copy of the image."""

    result = image.copy()

    cv2.drawContours(
        result,
        contours,
        -1,
        (0, 255, 0),
        2
    )

    return result
