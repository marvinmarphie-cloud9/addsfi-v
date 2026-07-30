from pathlib import Path
import copy

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


TRAIN_DIR = Path("datasets/train")
VAL_DIR = Path("datasets/val")
MODEL_OUTPUT = Path("models/trained/deepfake_model.pt")

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.001
EXPECTED_CLASSES = {
    "0_real": 0,
    "1_fake": 1,
}


def create_dataloaders() -> tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")

    if not VAL_DIR.exists():
        raise FileNotFoundError(f"Validation directory not found: {VAL_DIR}")

    weights = EfficientNet_B0_Weights.DEFAULT

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    validation_transform = weights.transforms()

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform,
    )

    validation_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=validation_transform,
    )

    if train_dataset.class_to_idx != EXPECTED_CLASSES:
        raise ValueError(
            "Incorrect training folders. Expected: "
            f"{EXPECTED_CLASSES}, found: {train_dataset.class_to_idx}"
        )

    if validation_dataset.class_to_idx != EXPECTED_CLASSES:
        raise ValueError(
            "Incorrect validation folders. Expected: "
            f"{EXPECTED_CLASSES}, found: {validation_dataset.class_to_idx}"
        )

    if len(train_dataset) == 0:
        raise ValueError("The training dataset is empty.")

    if len(validation_dataset) == 0:
        raise ValueError("The validation dataset is empty.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")
    print(f"Classes: {train_dataset.class_to_idx}")

    return train_loader, validation_loader


def create_model(device: torch.device) -> nn.Module:
    """Create an EfficientNet-B0 binary classifier."""

    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, 2)

    return model.to(device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> tuple[float, float]:
    """Run one training or validation epoch."""

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_function(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        correct_predictions += (
            predictions == labels
        ).sum().item()
        total_samples += labels.size(0)

    average_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


def train() -> None:
    """Train the classifier and save the best validation model."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    train_loader, validation_loader = create_dataloaders()
    model = create_model(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_accuracy = 0.0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=train_loader,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
        )

        validation_loss, validation_accuracy = run_epoch(
            model=model,
            loader=validation_loader,
            loss_function=loss_function,
            device=device,
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.4f} | "
            f"val loss: {validation_loss:.4f} | "
            f"val accuracy: {validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_model_state = copy.deepcopy(model.state_dict())

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        best_model_state,
        MODEL_OUTPUT,
    )

    print()
    print("Training complete.")
    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.4f}"
    )
    print(f"Model saved to: {MODEL_OUTPUT}")


if __name__ == "__main__":
    train()