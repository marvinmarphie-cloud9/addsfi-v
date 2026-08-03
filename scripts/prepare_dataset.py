from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from addfs.config import (
    FACEFORENSICS_ROOT,
    FRAME_INTERVAL,
    PREPARED_DATASET,
    WORKING_DATASET,
    validate_configuration,
)
from addfs.dataset_builder import DatasetBuilder


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
}


def get_video_paths(
    video_folder: Path,
) -> list[Path]:
    if not video_folder.exists():
        raise FileNotFoundError(
            f"Video folder does not exist: {video_folder}"
        )

    return sorted(
        path
        for path in video_folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def split_videos(
    video_paths: list[Path],
    validation_ratio: float = 0.2,
) -> tuple[list[Path], list[Path]]:
    if not video_paths:
        return [], []

    if len(video_paths) == 1:
        return video_paths, []

    validation_count = max(
        1,
        int(len(video_paths) * validation_ratio),
    )

    validation_count = min(
        validation_count,
        len(video_paths) - 1,
    )

    training_videos = video_paths[:-validation_count]
    validation_videos = video_paths[-validation_count:]

    return training_videos, validation_videos


def process_video_group(
    builder: DatasetBuilder,
    video_paths: list[Path],
    class_name: str,
    split_name: str,
) -> tuple[int, int]:
    total_training_faces = 0
    total_validation_faces = 0

    if not video_paths:
        print(f"No {split_name} videos to process.")
        return total_training_faces, total_validation_faces

    print(f"\n{split_name.upper()} videos")

    for index, video_path in enumerate(
        video_paths,
        start=1,
    ):
        print(
            f"[{index}/{len(video_paths)}] "
            f"{video_path.name}"
        )

        try:
            train_paths, validation_paths = builder.prepare(
                video_path=video_path,
                class_name=class_name,
                split_name=split_name,
            )

            total_training_faces += len(train_paths)
            total_validation_faces += len(
                validation_paths
            )

            print(
                f"Saved {len(train_paths)} training faces "
                f"and {len(validation_paths)} "
                "validation faces."
            )

        except Exception as error:
            print(
                f"Failed to process "
                f"{video_path.name}: {error}"
            )

    return (
        total_training_faces,
        total_validation_faces,
    )


def process_folder(
    builder: DatasetBuilder,
    video_folder: Path,
    class_name: str,
    validation_ratio: float = 0.2,
) -> tuple[int, int]:
    video_paths = get_video_paths(video_folder)

    if not video_paths:
        print(f"No videos found in: {video_folder}")
        return 0, 0

    training_videos, validation_videos = split_videos(
        video_paths=video_paths,
        validation_ratio=validation_ratio,
    )

    print(
        f"\nProcessing {len(video_paths)} videos "
        f"as '{class_name}'"
    )

    print(
        f"Video split: {len(training_videos)} train, "
        f"{len(validation_videos)} validation"
    )

    training_faces, _ = process_video_group(
        builder=builder,
        video_paths=training_videos,
        class_name=class_name,
        split_name="train",
    )

    _, validation_faces = process_video_group(
        builder=builder,
        video_paths=validation_videos,
        class_name=class_name,
        split_name="val",
    )

    print(
        f"\nFinished '{class_name}': "
        f"{training_faces} training faces, "
        f"{validation_faces} validation faces."
    )

    return training_faces, validation_faces


def main() -> None:
    warnings = validate_configuration()

    for warning in warnings:
        print(f"Configuration warning: {warning}")

    real_folder = (
        FACEFORENSICS_ROOT
        / "original_sequences"
        / "youtube"
        / "c23"
        / "videos"
    )

    fake_folder = (
        FACEFORENSICS_ROOT
        / "manipulated_sequences"
        / "Deepfakes"
        / "c23"
        / "videos"
    )

    frame_output_directory = (
        WORKING_DATASET / "frames"
    )

    face_output_directory = (
        WORKING_DATASET / "faces"
    )

    print("Dataset configuration")
    print(f"FaceForensics++ root: {FACEFORENSICS_ROOT}")
    print(f"Real videos: {real_folder}")
    print(f"Fake videos: {fake_folder}")
    print(f"Working directory: {WORKING_DATASET}")
    print(f"Prepared dataset: {PREPARED_DATASET}")
    print(f"Frame interval: {FRAME_INTERVAL}")

    builder = DatasetBuilder(
        frame_output_dir=frame_output_directory,
        face_output_dir=face_output_directory,
        dataset_output_dir=PREPARED_DATASET,
        frame_interval=FRAME_INTERVAL,
    )

    real_train, real_validation = process_folder(
        builder=builder,
        video_folder=real_folder,
        class_name="0_real",
        validation_ratio=0.2,
    )

    fake_train, fake_validation = process_folder(
        builder=builder,
        video_folder=fake_folder,
        class_name="1_fake",
        validation_ratio=0.2,
    )

    total_faces = (
        real_train
        + real_validation
        + fake_train
        + fake_validation
    )

    print("\nDataset preparation complete.")
    print("\nFinal summary")
    print(f"Real training faces: {real_train}")
    print(
        f"Real validation faces: "
        f"{real_validation}"
    )
    print(f"Fake training faces: {fake_train}")
    print(
        f"Fake validation faces: "
        f"{fake_validation}"
    )
    print(f"Total faces: {total_faces}")


if __name__ == "__main__":
    main()