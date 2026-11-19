
import cv2
import numpy as np


def validate_image(image):
    """Check whether an image is valid."""
    if image is None:
        raise ValueError("Image cannot be empty or None.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Image cannot be empty.")

    if image.ndim not in (2, 3):
        raise ValueError("Image must have 2 or 3 dimensions.")

    return True


def convert_bgr_to_rgb(image):
    """Convert an OpenCV BGR image to RGB."""
    validate_image(image)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    raise ValueError("Unsupported number of image channels.")


def get_image_dimensions(image):
    """Return image width, height, and channel count."""
    validate_image(image)

    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]

    return {
        "width": width,
        "height": height,
        "channels": channels
    }
