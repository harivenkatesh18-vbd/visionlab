
from ultralytics import YOLO

_model = None


def load_model():
    """Load the pretrained YOLO model once."""
    global _model

    if _model is None:
        _model = YOLO("yolov8n.pt")

    return _model


def detect_objects(image, confidence=0.25):
    """
    Detect objects in an image.

    Returns:
        annotated_image: Image with bounding boxes.
        detections: List of detected objects and confidence scores.
    """

    if image is None or image.size == 0:
        raise ValueError("Input image cannot be empty.")

    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1.")

    model = load_model()

    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False,
    )

    result = results[0]
    annotated_image = result.plot()

    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        score = float(box.conf[0])

        detections.append({
            "name": result.names[class_id],
            "confidence": round(score, 3),
        })

    return annotated_image, detections
