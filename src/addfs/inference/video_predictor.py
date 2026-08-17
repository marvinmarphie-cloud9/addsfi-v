from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from torch import nn

from addfs.dataset_builder import FaceProcessor, FrameExtractor
from addfs.inference.image_predictor import ImagePredictor


class VideoPredictor:
    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: Path,
        frame_interval: int = 30,
        threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.frame_interval = frame_interval
        self.threshold = threshold

    def predict(
        self,
        video_path: Path,
    ) -> dict[str, Any]:
        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        start_time = perf_counter()

        with TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)

            frame_directory = temporary_root / "frames"
            face_directory = temporary_root / "faces"

            frame_extractor = FrameExtractor(
                output_dir=frame_directory,
                frame_interval=self.frame_interval,
            )

            face_processor = FaceProcessor(
                output_dir=face_directory,
            )

            frame_paths = frame_extractor.extract(video_path)

            face_paths = face_processor.process(
                frame_paths
            )

            if not face_paths:
                raise ValueError(
                    "No faces were detected in the video."
                )

            image_predictor = ImagePredictor(
                model=self.model,
                checkpoint_path=self.checkpoint_path,
            )

            face_results = [
                image_predictor.predict(face_path)
                for face_path in face_paths
            ]

            average_fake_probability = sum(
                result["fake_probability"]
                for result in face_results
            ) / len(face_results)

            average_real_probability = (
                1.0 - average_fake_probability
            )

            prediction = (
                "1_fake"
                if average_fake_probability
                >= self.threshold
                else "0_real"
            )

            confidence = max(
                average_real_probability,
                average_fake_probability,
            )

            processing_seconds = (
                perf_counter() - start_time
            )

            return {
                "video": str(video_path),
                "prediction": prediction,
                "confidence": confidence,
                "real_probability": (
                    average_real_probability
                ),
                "fake_probability": (
                    average_fake_probability
                ),
                "frames_analyzed": len(frame_paths),
                "faces_analyzed": len(face_paths),
                "threshold": self.threshold,
                "processing_seconds": round(
                    processing_seconds,
                    3,
                ),
                "model_name": "EfficientNet-B0",
                "checkpoint": (
                    self.checkpoint_path.name
                ),
                "frame_interval": (
                    self.frame_interval
                ),
            }