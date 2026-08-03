from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from addfs.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    NUM_WORKERS,
    PREPARED_DATASET,
    SCHEDULER_PATIENCE,
    create_runtime_directories,
)
from addfs.training import Trainer, create_dataloaders


def validate_dataset_directories(
    training_directory: Path,
    validation_directory: Path,
) -> None:
    required_directories = [
        training_directory / "0_real",
        training_directory / "1_fake",
        validation_directory / "0_real",
        validation_directory / "1_fake",
    ]

    missing_directories = [
        directory
        for directory in required_directories
        if not directory.exists()
    ]

    if missing_directories:
        missing_text = "\n".join(
            str(directory)
            for directory in missing_directories
        )

        raise FileNotFoundError(
            "Prepared dataset directories are missing:\n"
            f"{missing_text}\n\n"
            "Run scripts\\prepare_dataset.py first."
        )


def count_images(directory: Path) -> int:
    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    )


def main() -> None:
    create_runtime_directories()

    training_directory = (
        PREPARED_DATASET / "train"
    )

    validation_directory = (
        PREPARED_DATASET / "val"
    )

    validate_dataset_directories(
        training_directory=training_directory,
        validation_directory=validation_directory,
    )

    training_image_count = count_images(
        training_directory
    )

    validation_image_count = count_images(
        validation_directory
    )

    if training_image_count == 0:
        raise ValueError(
            "The training dataset contains no images."
        )

    if validation_image_count == 0:
        raise ValueError(
            "The validation dataset contains no images."
        )

    device_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    print("Training configuration")
    print(f"Device: {device_name}")
    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )
    print(
        f"Training directory: "
        f"{training_directory}"
    )
    print(
        f"Validation directory: "
        f"{validation_directory}"
    )
    print(
        f"Training images: "
        f"{training_image_count}"
    )
    print(
        f"Validation images: "
        f"{validation_image_count}"
    )
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(
        "Early-stopping patience: "
        f"{EARLY_STOPPING_PATIENCE}"
    )
    print(
        "Scheduler patience: "
        f"{SCHEDULER_PATIENCE}"
    )

    train_loader, validation_loader = (
        create_dataloaders(
            training_directory,
            validation_directory,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
    )

    trainer = Trainer(
        train_loader=train_loader,
        validation_loader=validation_loader,
        learning_rate=LEARNING_RATE,
        patience=EARLY_STOPPING_PATIENCE,
        scheduler_patience=SCHEDULER_PATIENCE,
    )

    trainer.train(
        epochs=EPOCHS,
    )


if __name__ == "__main__":
    main()