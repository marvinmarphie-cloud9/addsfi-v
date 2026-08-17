from pathlib import Path
import random
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

VALIDATION_RATIO = 0.20
RANDOM_SEED = 42

# Real sources:
# 1000 YouTube originals + 363 actor originals.
REAL_SOURCES = [
    (
        "youtube",
        FACEFORENSICS_ROOT
        / "original_sequences"
        / "youtube"
        / "c23"
        / "videos",
    ),
    (
        "actors",
        FACEFORENSICS_ROOT
        / "original_sequences"
        / "actors"
        / "c23"
        / "videos",
    ),
]

# Equal contribution from each manipulation family.
FAKE_METHODS = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
    "FaceShifter",
    "DeepFakeDetection",
]

# 6 × 227 = 1362 fake videos.
# Real side contains approximately 1363 videos.
FAKE_VIDEOS_PER_METHOD = 227


def get_video_paths(
    video_folder: Path,
) -> list[Path]:
    if not video_folder.exists():
        print(
            f"Warning: video folder does not exist: "
            f"{video_folder}"
        )
        return []

    return sorted(
        path
        for path in video_folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def deterministic_sample(
    video_paths: list[Path],
    maximum_count: int | None,
    seed_offset: int = 0,
) -> list[Path]:
    if (
        maximum_count is None
        or len(video_paths) <= maximum_count
    ):
        return list(video_paths)

    random_generator = random.Random(
        RANDOM_SEED + seed_offset
    )

    sampled_paths = random_generator.sample(
        video_paths,
        maximum_count,
    )

    return sorted(sampled_paths)


def split_videos(
    video_paths: list[Path],
    validation_ratio: float = VALIDATION_RATIO,
    seed_offset: int = 0,
) -> tuple[list[Path], list[Path]]:
    if not video_paths:
        return [], []

    if len(video_paths) == 1:
        return video_paths, []

    shuffled_paths = list(video_paths)

    random_generator = random.Random(
        RANDOM_SEED + seed_offset
    )

    random_generator.shuffle(shuffled_paths)

    validation_count = max(
        1,
        int(len(shuffled_paths) * validation_ratio),
    )

    validation_count = min(
        validation_count,
        len(shuffled_paths) - 1,
    )

    validation_videos = shuffled_paths[
        :validation_count
    ]

    training_videos = shuffled_paths[
        validation_count:
    ]

    return (
        sorted(training_videos),
        sorted(validation_videos),
    )


def process_video_group(
    builder: DatasetBuilder,
    video_paths: list[Path],
    class_name: str,
    split_name: str,
    source_name: str,
) -> tuple[int, int]:
    total_training_faces = 0
    total_validation_faces = 0

    if not video_paths:
        print(
            f"No {split_name} videos for "
            f"{source_name}."
        )
        return 0, 0

    print(
        f"\n{source_name} — "
        f"{split_name.upper()}"
    )

    for index, video_path in enumerate(
        video_paths,
        start=1,
    ):
        print(
            f"[{index}/{len(video_paths)}] "
            f"{video_path.name}"
        )

        try:
            train_paths, validation_paths = (
                builder.prepare(
                    video_path=video_path,
                    class_name=class_name,
                    split_name=split_name,
                )
            )

            total_training_faces += len(
                train_paths
            )

            total_validation_faces += len(
                validation_paths
            )

            print(
                f"Saved {len(train_paths)} "
                "training faces and "
                f"{len(validation_paths)} "
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


def process_source(
    builder: DatasetBuilder,
    video_folder: Path,
    class_name: str,
    source_name: str,
    maximum_videos: int | None = None,
    seed_offset: int = 0,
) -> tuple[int, int, int]:
    available_videos = get_video_paths(
        video_folder
    )

    if not available_videos:
        print(
            f"\nNo videos found for "
            f"{source_name}: {video_folder}"
        )
        return 0, 0, 0

    selected_videos = deterministic_sample(
        video_paths=available_videos,
        maximum_count=maximum_videos,
        seed_offset=seed_offset,
    )

    (
        training_videos,
        validation_videos,
    ) = split_videos(
        video_paths=selected_videos,
        validation_ratio=VALIDATION_RATIO,
        seed_offset=seed_offset,
    )

    print(
        f"\n====================================="
    )
    print(f"Source: {source_name}")
    print(f"Class: {class_name}")
    print(
        f"Available videos: "
        f"{len(available_videos)}"
    )
    print(
        f"Selected videos: "
        f"{len(selected_videos)}"
    )
    print(
        f"Training videos: "
        f"{len(training_videos)}"
    )
    print(
        f"Validation videos: "
        f"{len(validation_videos)}"
    )
    print(
        f"====================================="
    )

    training_faces, _ = process_video_group(
        builder=builder,
        video_paths=training_videos,
        class_name=class_name,
        split_name="train",
        source_name=source_name,
    )

    _, validation_faces = process_video_group(
        builder=builder,
        video_paths=validation_videos,
        class_name=class_name,
        split_name="val",
        source_name=source_name,
    )

    return (
        training_faces,
        validation_faces,
        len(selected_videos),
    )


def count_images(
    directory: Path,
) -> int:
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    if not directory.exists():
        return 0

    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in image_extensions
    )


def main() -> None:
    warnings = validate_configuration()

    for warning in warnings:
        print(
            f"Configuration warning: "
            f"{warning}"
        )

    print("\nADDFS Multi-Manipulation Dataset")
    print("================================")
    print(
        f"FaceForensics++ root: "
        f"{FACEFORENSICS_ROOT}"
    )
    print(
        f"Working dataset: "
        f"{WORKING_DATASET}"
    )
    print(
        f"Prepared dataset: "
        f"{PREPARED_DATASET}"
    )
    print(
        f"Frame interval: "
        f"{FRAME_INTERVAL}"
    )
    print(
        f"Validation ratio: "
        f"{VALIDATION_RATIO:.0%}"
    )
    print(
        f"Random seed: "
        f"{RANDOM_SEED}"
    )
    print(
        "Fake videos per method: "
        f"{FAKE_VIDEOS_PER_METHOD}"
    )

    frame_output_directory = (
        WORKING_DATASET / "frames"
    )

    face_output_directory = (
        WORKING_DATASET / "faces"
    )

    builder = DatasetBuilder(
        frame_output_dir=frame_output_directory,
        face_output_dir=face_output_directory,
        dataset_output_dir=PREPARED_DATASET,
        frame_interval=FRAME_INTERVAL,
    )

    total_real_training_faces = 0
    total_real_validation_faces = 0
    total_real_videos = 0

    print("\n\nREAL DATA")
    print("=========")

    for index, (
        source_name,
        source_folder,
    ) in enumerate(
        REAL_SOURCES,
        start=1,
    ):
        (
            training_faces,
            validation_faces,
            selected_count,
        ) = process_source(
            builder=builder,
            video_folder=source_folder,
            class_name="0_real",
            source_name=(
                f"original_{source_name}"
            ),
            maximum_videos=None,
            seed_offset=index,
        )

        total_real_training_faces += (
            training_faces
        )

        total_real_validation_faces += (
            validation_faces
        )

        total_real_videos += selected_count

    total_fake_training_faces = 0
    total_fake_validation_faces = 0
    total_fake_videos = 0

    print("\n\nFAKE DATA")
    print("=========")

    for index, method_name in enumerate(
        FAKE_METHODS,
        start=100,
    ):
        method_folder = (
            FACEFORENSICS_ROOT
            / "manipulated_sequences"
            / method_name
            / "c23"
            / "videos"
        )

        (
            training_faces,
            validation_faces,
            selected_count,
        ) = process_source(
            builder=builder,
            video_folder=method_folder,
            class_name="1_fake",
            source_name=method_name,
            maximum_videos=(
                FAKE_VIDEOS_PER_METHOD
            ),
            seed_offset=index,
        )

        total_fake_training_faces += (
            training_faces
        )

        total_fake_validation_faces += (
            validation_faces
        )

        total_fake_videos += selected_count

    training_real_directory = (
        PREPARED_DATASET
        / "train"
        / "0_real"
    )

    training_fake_directory = (
        PREPARED_DATASET
        / "train"
        / "1_fake"
    )

    validation_real_directory = (
        PREPARED_DATASET
        / "val"
        / "0_real"
    )

    validation_fake_directory = (
        PREPARED_DATASET
        / "val"
        / "1_fake"
    )

    actual_real_train = count_images(
        training_real_directory
    )

    actual_fake_train = count_images(
        training_fake_directory
    )

    actual_real_validation = count_images(
        validation_real_directory
    )

    actual_fake_validation = count_images(
        validation_fake_directory
    )

    total_faces = (
        actual_real_train
        + actual_fake_train
        + actual_real_validation
        + actual_fake_validation
    )

    print(
        "\n\n====================================="
    )
    print(
        "MULTI-MANIPULATION DATASET COMPLETE"
    )
    print(
        "====================================="
    )

    print("\nSource videos")
    print(
        f"Real videos: "
        f"{total_real_videos}"
    )
    print(
        f"Fake videos: "
        f"{total_fake_videos}"
    )

    print("\nTraining faces")
    print(
        f"Real: "
        f"{actual_real_train}"
    )
    print(
        f"Fake: "
        f"{actual_fake_train}"
    )

    print("\nValidation faces")
    print(
        f"Real: "
        f"{actual_real_validation}"
    )
    print(
        f"Fake: "
        f"{actual_fake_validation}"
    )

    print(
        f"\nTotal prepared faces: "
        f"{total_faces}"
    )

    if actual_real_train > 0:
        training_ratio = (
            actual_fake_train
            / actual_real_train
        )

        print(
            "Training fake/real ratio: "
            f"{training_ratio:.2f}"
        )

    if actual_real_validation > 0:
        validation_ratio = (
            actual_fake_validation
            / actual_real_validation
        )

        print(
            "Validation fake/real ratio: "
            f"{validation_ratio:.2f}"
        )

    print(
        "\nPrepared dataset location:"
    )
    print(PREPARED_DATASET)

    print(
        "\nDo NOT start training until "
        "the class counts above have "
        "been reviewed."
    )


if __name__ == "__main__":
    main()
