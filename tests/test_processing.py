from modules.contour_detection import detect_contours, draw_contours
from modules.shape_detection import detect_shapes, classify_shape
from modules.color_detection import create_color_mask, extract_color
from modules.edge_detection import detect_edges
import numpy as np

from modules.image_processing import (
    convert_to_grayscale,
    apply_gaussian_blur,
    apply_median_blur,
    sharpen_image,
    adjust_brightness_contrast,
    apply_threshold,
    resize_image
)


def create_test_image():
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    image[50:250, 50:350] = (255, 255, 255)
    return image


def test_grayscale():
    image = create_test_image()
    result = convert_to_grayscale(image)
    assert len(result.shape) == 2


def test_gaussian_blur():
    image = create_test_image()
    result = apply_gaussian_blur(image)
    assert result.shape == image.shape


def test_median_blur():
    image = create_test_image()
    result = apply_median_blur(image)
    assert result.shape == image.shape


def test_sharpen():
    image = create_test_image()
    result = sharpen_image(image)
    assert result.shape == image.shape


def test_brightness_contrast():
    image = create_test_image()
    result = adjust_brightness_contrast(image, 20, 1.2)
    assert result.shape == image.shape


def test_threshold():
    image = create_test_image()
    result = apply_threshold(image)
    assert len(result.shape) == 2


def test_resize():
    image = create_test_image()
    result = resize_image(image, 200)
    assert result.shape[1] == 200
def test_edge_detection():
    image = create_test_image()
    result = detect_edges(image)

    assert len(result.shape) == 2
    assert result.shape == image.shape[:2]


def test_contour_detection():
    image = create_test_image()
    contours = detect_contours(image)
    assert len(contours) > 0


def test_draw_contours():
    image = create_test_image()
    contours = detect_contours(image)
    result = draw_contours(image, contours)
    assert result.shape == image.shape


def test_shape_detection():
    image = create_test_image()
    result, shapes = detect_shapes(image)

    assert result.shape == image.shape
    assert isinstance(shapes, list)

def test_color_mask():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:] = (0, 0, 255)  # Red in BGR

    mask = create_color_mask(image, [0, 100, 100], [10, 255, 255])

    assert mask.shape == (100, 100)
    assert np.all(mask == 255)


def test_extract_color():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.uint8) * 255

    result = extract_color(image, mask)

    assert result.shape == image.shape