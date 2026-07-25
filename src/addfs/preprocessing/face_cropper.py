from pathlib import Path
from uuid import uuid4

import cv2


def crop_faces(
    image_path: str | Path,
    faces: list[tuple[int, int, int, int]],
    output_root: str | Path = "temp/faces",
    target_size: tuple[int, int] = (224, 224),
    padding_ratio: float = 0.15,
) -> list[Path]:
    """Crop, resize, and save detected faces from one frame."""

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Frame not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read frame: {image_path}")

    image_height, image_width = image.shape[:2]

    output_dir = Path(output_root) / uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=True)

    cropped_paths: list[Path] = []

    for index, (x, y, width, height) in enumerate(faces):
        padding_x = int(width * padding_ratio)
        padding_y = int(height * padding_ratio)

        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(image_width, x + width + padding_x)
        y2 = min(image_height, y + height + padding_y)

        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        resized_face = cv2.resize(face_crop, target_size)

        crop_path = output_dir / f"face_{index:03d}.jpg"

        if cv2.imwrite(str(crop_path), resized_face):
            cropped_paths.append(crop_path)

    return cropped_paths