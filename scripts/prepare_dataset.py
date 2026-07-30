from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from addfs.dataset_builder import DatasetBuilder


def get_video_paths(video_folder: Path) -> list[Path]:
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    if not video_folder.exists():
        raise FileNotFoundError(
            f"Video folder does not exist: {video_folder}"
        )

    return sorted(
        path
        for path in video_folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in video_extensions
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

    for index, video_path in enumerate(video_paths, start=1):
        print(
            f"[{index}/{len(video_paths)}] "
            f"{video_path.name}"
        )

        try:
            train_paths, val_paths = builder.prepare(
                video_path=video_path,
                class_name=class_name,
                split_name=split_name,
            )

            total_training_faces += len(train_paths)
            total_validation_faces += len(val_paths)

            print(
                f"Saved {len(train_paths)} training faces "
                f"and {len(val_paths)} validation faces."
            )

        except Exception as error:
            print(
                f"Failed to process "
                f"{video_path.name}: {error}"
            )

    return total_training_faces, total_validation_faces


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
    real_folder = (
        PROJECT_ROOT
        / "datasets"
        / "faceforensics"
        / "original_sequences"
        / "youtube"
        / "c23"
        / "videos"
    )

    fake_folder = (
        PROJECT_ROOT
        / "datasets"
        / "faceforensics"
        / "manipulated_sequences"
        / "Deepfakes"
        / "c23"
        / "videos"
    )

    builder = DatasetBuilder(
        frame_output_dir=(
            PROJECT_ROOT
            / "datasets"
            / "working"
            / "frames"
        ),
        face_output_dir=(
            PROJECT_ROOT
            / "datasets"
            / "working"
            / "faces"
        ),
        dataset_output_dir=(
            PROJECT_ROOT
            / "datasets"
            / "prepared"
        ),
        frame_interval=30,
    )

    real_train, real_val = process_folder(
        builder=builder,
        video_folder=real_folder,
        class_name="0_real",
        validation_ratio=0.2,
    )

    fake_train, fake_val = process_folder(
        builder=builder,
        video_folder=fake_folder,
        class_name="1_fake",
        validation_ratio=0.2,
    )

    print("\nDataset preparation complete.")
    print("\nFinal summary")
    print(f"Real training faces: {real_train}")
    print(f"Real validation faces: {real_val}")
    print(f"Fake training faces: {fake_train}")
    print(f"Fake validation faces: {fake_val}")
    print(
        "Total faces: "
        f"{real_train + real_val + fake_train + fake_val}"
    )


if __name__ == "__main__":
    main()