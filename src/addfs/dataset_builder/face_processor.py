from pathlib import Path

import cv2


class FaceProcessor:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.face_detector = cv2.CascadeClassifier(cascade_path)

        if self.face_detector.empty():
            raise RuntimeError("Could not load the face detector.")

    def process(self, frame_paths: list[Path]) -> list[Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        saved_faces: list[Path] = []

        for frame_path in frame_paths:
            image = cv2.imread(str(frame_path))

            if image is None:
                print(f"Skipped unreadable frame: {frame_path.name}")
                continue

            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = self.face_detector.detectMultiScale(
                grayscale,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )

            for face_number, (x, y, width, height) in enumerate(faces):
                face = image[y : y + height, x : x + width]

                output_path = self.output_dir / (
                    f"{frame_path.stem}_face_{face_number:02d}.jpg"
                )

                if cv2.imwrite(str(output_path), face):
                    saved_faces.append(output_path)

        return saved_faces