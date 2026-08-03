from pathlib import Path
import sys

import torch
from torch import nn
from torchvision.models import efficientnet_b0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from addfs.evaluation import Evaluator
from addfs.training import create_dataloaders


def find_latest_checkpoint() -> Path:
    checkpoints = sorted(
        PROJECT_ROOT.glob(
            "results/training/run_*/best_model.pt"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not checkpoints:
        raise FileNotFoundError(
            "No checkpoint found under results/training."
        )

    return checkpoints[0]


def load_model(checkpoint_path: Path) -> nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model = efficientnet_b0(weights=None)

    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, 2)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model


def main() -> None:
    checkpoint_path = find_latest_checkpoint()

    _, validation_loader = create_dataloaders(
        train_directory=PROJECT_ROOT / "datasets/prepared/train",
        validation_directory=PROJECT_ROOT / "datasets/prepared/val",
        batch_size=8,
        num_workers=0,
    )

    output_directory = (
        checkpoint_path.parent / "evaluation"
    )

    evaluator = Evaluator(
        model=load_model(checkpoint_path),
        validation_loader=validation_loader,
        output_directory=output_directory,
    )

    metrics = evaluator.evaluate()

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 score:  {metrics['f1_score']:.2%}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"Results:   {output_directory}")


if __name__ == "__main__":
    main()