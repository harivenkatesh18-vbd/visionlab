
import numpy as np
import pytest
from types import SimpleNamespace

from modules import object_detection


def test_empty_image():
    with pytest.raises(ValueError):
        object_detection.detect_objects(
            np.array([]),
        )


def test_invalid_confidence():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        object_detection.detect_objects(
            image,
            confidence=1.5,
        )


def test_multiple_object_detection(monkeypatch):
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    fake_boxes = [
        SimpleNamespace(
            cls=np.array([0]),
            conf=np.array([0.95]),
        ),
        SimpleNamespace(
            cls=np.array([2]),
            conf=np.array([0.88]),
        ),
    ]

    fake_result = SimpleNamespace(
        boxes=fake_boxes,
        names={
            0: "person",
            2: "car",
        },
        plot=lambda: image.copy(),
    )

    fake_model = SimpleNamespace(
        predict=lambda **kwargs: [fake_result]
    )

    monkeypatch.setattr(
        object_detection,
        "load_model",
        lambda: fake_model,
    )

    processed, detections = object_detection.detect_objects(
        image,
        confidence=0.25,
    )

    assert processed.shape == image.shape
    assert len(detections) == 2
    assert detections[0]["name"] == "person"
    assert detections[1]["name"] == "car"
