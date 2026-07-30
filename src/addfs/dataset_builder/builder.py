from pathlib import Path

from .frame_extractor import FrameExtractor
from .face_processor import FaceProcessor
from .splitter import DatasetSplitter


class DatasetBuilder:
    def __init__(
        self,
        frame_output_dir: Path,
        face_output_dir: Path,
        dataset_output_dir: Path,
        frame_interval: int = 30,
    ) -> None:
        self.frame_extractor = FrameExtractor(
            output_dir=frame_output_dir,
            frame_interval=frame_interval,
        )

        self.face_processor = FaceProcessor(
            output_dir=face_output_dir,
        )

        self.splitter = DatasetSplitter(
            output_dir=dataset_output_dir,
        )

    def prepare(
        self,
        video_path: Path,
        class_name: str,
        split_name: str | None = None,
    ) -> tuple[list[Path], list[Path]]:
        frame_paths = self.frame_extractor.extract(
            video_path
        )

        face_paths = self.face_processor.process(
            frame_paths
        )

        train_paths, val_paths = self.splitter.split(
            image_paths=face_paths,
            class_name=class_name,
            split_name=split_name,
            source_name=video_path.stem,
        )

        return train_paths, val_paths