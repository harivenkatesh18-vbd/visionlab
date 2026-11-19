
import cv2
from collections import deque


def process_video(input_path, output_path, operation="grayscale"):
    """Process a video frame by frame."""

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError("Unable to open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video properties.")

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError("Unable to create output video.")

    trail = deque(maxlen=30)

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            if operation == "grayscale":
                gray = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2GRAY
                )
                processed = cv2.cvtColor(
                    gray, cv2.COLOR_GRAY2BGR
                )

            elif operation == "edges":
                gray = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2GRAY
                )
                edges = cv2.Canny(gray, 100, 200)
                processed = cv2.cvtColor(
                    edges, cv2.COLOR_GRAY2BGR
                )

            elif operation == "blur":
                processed = cv2.GaussianBlur(
                    frame, (15, 15), 0
                )

            elif operation == "tracking":
                processed = frame.copy()

                hsv = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2HSV
                )

                lower_red1 = (0, 100, 100)
                upper_red1 = (10, 255, 255)

                lower_red2 = (170, 100, 100)
                upper_red2 = (180, 255, 255)

                mask1 = cv2.inRange(
                    hsv, lower_red1, upper_red1
                )

                mask2 = cv2.inRange(
                    hsv, lower_red2, upper_red2
                )

                mask = cv2.bitwise_or(mask1, mask2)

                contours, _ = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                if contours:
                    largest = max(
                        contours,
                        key=cv2.contourArea
                    )

                    if cv2.contourArea(largest) > 300:
                        x, y, w, h = cv2.boundingRect(
                            largest
                        )

                        center = (
                            x + w // 2,
                            y + h // 2
                        )

                        trail.append(center)

                        cv2.rectangle(
                            processed,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            processed,
                            "Red Object",
                            (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

                for i in range(1, len(trail)):
                    cv2.line(
                        processed,
                        trail[i - 1],
                        trail[i],
                        (255, 0, 0),
                        2
                    )

            else:
                raise ValueError(
                    "Unsupported video operation."
                )

            writer.write(processed)

    finally:
        cap.release()
        writer.release()

    return output_path
