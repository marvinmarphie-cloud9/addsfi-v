from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets, transforms


def create_image_transform(
    image_size: int = 224,
    training: bool = False,
) -> transforms.Compose:
    transform_steps = [
        transforms.Resize((image_size, image_size)),
    ]

    if training:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                ),
            ]
        )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transforms.Compose(transform_steps)


def create_image_dataset(
    dataset_directory: Path,
    image_size: int = 224,
    training: bool = False,
) -> Dataset:
    if not dataset_directory.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: "
            f"{dataset_directory}"
        )

    dataset = datasets.ImageFolder(
        root=dataset_directory,
        transform=create_image_transform(
            image_size=image_size,
            training=training,
        ),
    )

    if not dataset.samples:
        raise ValueError(
            f"No images were found in: {dataset_directory}"
        )

    return dataset