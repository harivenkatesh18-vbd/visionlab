
import cv2
import numpy as np
import pytest

from modules.video_processing import process_video


def create_sample_video(path):
    """Create a small synthetic video for testing."""
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (100, 100)
    )

    for _ in range(10):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:] = (0, 100, 200)
        writer.write(frame)

    writer.release()


def test_grayscale_video_processing(tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"

    create_sample_video(input_path)

    result = process_video(
        str(input_path),
        str(output_path),
        operation="grayscale"
    )

    assert result == str(output_path)
    assert output_path.exists()

    cap = cv2.VideoCapture(str(output_path))
    success, frame = cap.read()
    cap.release()

    assert success
    assert np.allclose(frame[:, :, 0], frame[:, :, 1], atol=3)
    assert np.allclose(frame[:, :, 1], frame[:, :, 2], atol=3)


def test_invalid_video_path(tmp_path):
    output_path = tmp_path / "output.mp4"

    with pytest.raises(ValueError):
        process_video(
            str(tmp_path / "missing.mp4"),
            str(output_path)
        )


def test_unsupported_operation(tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"

    create_sample_video(input_path)

    with pytest.raises(ValueError):
        process_video(
            str(input_path),
            str(output_path),
            operation="invalid"
        )
