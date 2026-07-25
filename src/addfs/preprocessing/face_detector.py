from pathlib import Path

import cv2


class FaceDetector:
    """Detect faces in image files using OpenCV Haar cascades."""

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

        if self.detector.empty():
            raise RuntimeError("OpenCV face detector could not be loaded.")

    def detect_faces(self, image_path: str | Path) -> list[tuple[int, int, int, int]]:
        """
        Detect faces and return bounding boxes as:
        (x, y, width, height)
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            grayscale,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        return [tuple(map(int, face)) for face in faces]