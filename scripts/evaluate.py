from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch import nn
from torchvision.models import efficientnet_b0

from addfs.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    PREPARED_DATASET,
    RESULTS_DIR,
)
from addfs.evaluation import Evaluator
from addfs.training import create_dataloaders


def find_latest_checkpoint() -> Path:
    checkpoints = sorted(
        RESULTS_DIR.glob(
            "training/run_*/best_model.pt"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not checkpoints:
        raise FileNotFoundError(
            "No checkpoint found under "
            f"{RESULTS_DIR / 'training'}."
        )

    return checkpoints[0]


def load_model(
    checkpoint_path: Path,
) -> nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model = efficientnet_b0(weights=None)

    input_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = nn.Linear(
        input_features,
        2,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model


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


def main() -> None:
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

    checkpoint_path = find_latest_checkpoint()

    _, validation_loader = create_dataloaders(
        train_directory=training_directory,
        validation_directory=validation_directory,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    output_directory = (
        checkpoint_path.parent / "evaluation"
    )

    print("Evaluation configuration")
    print(f"Checkpoint: {checkpoint_path}")
    print(
        f"Validation directory: "
        f"{validation_directory}"
    )
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Output: {output_directory}")

    evaluator = Evaluator(
        model=load_model(checkpoint_path),
        validation_loader=validation_loader,
        output_directory=output_directory,
    )

    metrics = evaluator.evaluate()

    print("\nEvaluation complete")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 score:  {metrics['f1_score']:.2%}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"Results:   {output_directory}")


if __name__ == "__main__":
    main()