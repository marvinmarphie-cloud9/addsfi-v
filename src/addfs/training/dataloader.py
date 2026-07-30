from pathlib import Path

from torch.utils.data import DataLoader

from .dataset import create_image_dataset


def create_dataloaders(
    train_directory: Path,
    validation_directory: Path,
    batch_size: int = 16,
    num_workers: int = 0,
):
    train_dataset = create_image_dataset(
        train_directory,
        training=True,
    )

    validation_dataset = create_image_dataset(
        validation_directory,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return (
        train_loader,
        validation_loader,
    )