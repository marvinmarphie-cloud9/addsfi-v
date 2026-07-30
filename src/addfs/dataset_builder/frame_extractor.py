from pathlib import Path

import cv2


class FrameExtractor:
    def __init__(self, output_dir: Path, frame_interval: int = 30) -> None:
        self.output_dir = output_dir
        self.frame_interval = frame_interval

    def extract(self, video_path: Path) -> list[Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        saved_frames: list[Path] = []

        frame_number = 0
        saved_number = 0

        while True:
            success, frame = capture.read()

            if not success:
                break

            if frame_number % self.frame_interval == 0:
                output_path = (
                    self.output_dir
                    / f"{video_path.stem}_frame_{saved_number:04d}.jpg"
                )

                if cv2.imwrite(str(output_path), frame):
                    saved_frames.append(output_path)
                    saved_number += 1

            frame_number += 1

        capture.release()

        return saved_frames

