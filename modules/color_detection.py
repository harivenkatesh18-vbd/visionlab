
import cv2
import numpy as np


def create_color_mask(image, lower_hsv, upper_hsv):
    """Create a mask for a selected HSV color range."""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)

    return cv2.inRange(hsv, lower, upper)


def extract_color(image, mask):
    """Extract the selected color from the image."""

    return cv2.bitwise_and(image, image, mask=mask)
