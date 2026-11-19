
import cv2
import numpy as np


def convert_to_grayscale(image):
    """Convert a BGR image into grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(image, kernel_size=15):
    """Reduce image noise using Gaussian blur."""
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )


def apply_median_blur(image, kernel_size=5):
    """Remove noise while preserving edges."""
    return cv2.medianBlur(image, kernel_size)


def sharpen_image(image):
    """Enhance image details using a sharpening kernel."""
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(image, -1, kernel)


def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    """Adjust image brightness and contrast."""
    return cv2.convertScaleAbs(
        image,
        alpha=contrast,
        beta=brightness
    )


def apply_threshold(image, threshold_value=127):
    """Convert grayscale image into a binary image."""
    gray = convert_to_grayscale(image)

    _, result = cv2.threshold(
        gray,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    return result


def resize_image(image, width=800):
    """Resize image while maintaining aspect ratio."""
    height, original_width = image.shape[:2]

    new_height = int(height * width / original_width)

    return cv2.resize(
        image,
        (width, new_height),
        interpolation=cv2.INTER_AREA
    )
