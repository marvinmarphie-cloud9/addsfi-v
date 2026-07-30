from pathlib import Path
import random
import shutil


class DatasetSplitter:
    def __init__(
        self,
        output_dir: Path,
        validation_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.output_dir = output_dir
        self.validation_ratio = validation_ratio
        self.seed = seed

    def split(
        self,
        image_paths: list[Path],
        class_name: str,
        split_name: str | None = None,
        source_name: str | None = None,
    ) -> tuple[list[Path], list[Path]]:
        if not image_paths:
            return [], []

        if split_name is not None:
            if split_name not in {"train", "val"}:
                raise ValueError(
                    "split_name must be either 'train' or 'val'."
                )

            destination_dir = (
                self.output_dir
                / split_name
                / class_name
            )

            destination_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            copied_images = self._copy_images(
                image_paths=image_paths,
                destination_dir=destination_dir,
                source_name=source_name,
            )

            if split_name == "train":
                return copied_images, []

            return [], copied_images

        shuffled_paths = image_paths.copy()

        random_generator = random.Random(self.seed)
        random_generator.shuffle(shuffled_paths)

        validation_count = max(
            1,
            int(len(shuffled_paths) * self.validation_ratio),
        )

        validation_images = shuffled_paths[:validation_count]
        training_images = shuffled_paths[validation_count:]

        train_dir = self.output_dir / "train" / class_name
        validation_dir = self.output_dir / "val" / class_name

        train_dir.mkdir(parents=True, exist_ok=True)
        validation_dir.mkdir(parents=True, exist_ok=True)

        copied_training_images = self._copy_images(
            image_paths=training_images,
            destination_dir=train_dir,
            source_name=source_name,
        )

        copied_validation_images = self._copy_images(
            image_paths=validation_images,
            destination_dir=validation_dir,
            source_name=source_name,
        )

        return copied_training_images, copied_validation_images

    @staticmethod
    def _copy_images(
        image_paths: list[Path],
        destination_dir: Path,
        source_name: str | None = None,
    ) -> list[Path]:
        copied_images: list[Path] = []

        for index, image_path in enumerate(
            image_paths,
            start=1,
        ):
            prefix = source_name or image_path.parent.name

            destination_path = destination_dir / (
                f"{prefix}_face_{index:06d}"
                f"{image_path.suffix.lower()}"
            )

            duplicate_number = 1

            while destination_path.exists():
                destination_path = destination_dir / (
                    f"{prefix}_face_{index:06d}"
                    f"_{duplicate_number}"
                    f"{image_path.suffix.lower()}"
                )
                duplicate_number += 1

            shutil.copy2(
                image_path,
                destination_path,
            )

            copied_images.append(destination_path)

        return copied_images