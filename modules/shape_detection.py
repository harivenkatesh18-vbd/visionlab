
import cv2

from modules.contour_detection import detect_contours


def classify_shape(contour):
    """Classify a shape based on its contour."""

    perimeter = cv2.arcLength(contour, True)

    approximation = cv2.approxPolyDP(
        contour, 0.04 * perimeter, True
    )

    vertices = len(approximation)

    if vertices == 3:
        return "Triangle"

    elif vertices == 4:
        x, y, width, height = cv2.boundingRect(approximation)
        ratio = width / float(height)

        if 0.90 <= ratio <= 1.10:
            return "Square"

        return "Rectangle"

    elif vertices == 5:
        return "Pentagon"

    elif vertices > 7:
        return "Circle"

    return "Unknown"


def detect_shapes(image):
    """Detect and label shapes with improved label positioning."""

    contours = detect_contours(image)
    result = image.copy()
    detected_shapes = []

    for contour in contours:

        if cv2.contourArea(contour) < 300:
            continue

        shape = classify_shape(contour)

        x, y, width, height = cv2.boundingRect(contour)

        cv2.drawContours(
            result, [contour], -1, (0, 255, 0), 2
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(
            shape, font, font_scale, thickness
        )

        label_x = x + (width - text_width) // 2
        label_x = max(
            0,
            min(label_x, image.shape[1] - text_width - 4)
        )

        label_y = max(text_height + 5, y - 8)

        cv2.rectangle(
            result,
            (label_x, label_y - text_height - 4),
            (label_x + text_width + 4, label_y + baseline),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            result,
            shape,
            (label_x + 2, label_y),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA
        )

        detected_shapes.append(shape)

    return result, detected_shapes
