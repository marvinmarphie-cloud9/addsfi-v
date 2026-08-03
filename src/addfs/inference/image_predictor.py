from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn

from addfs.training import create_image_transform


class ImagePredictor:
    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: Path,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model = model.to(self.device)
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.model.eval()

        self.class_names = checkpoint.get(
            "class_names",
            ["0_real", "1_fake"],
        )

        self.transform = create_image_transform(
            image_size=224,
            training=False,
        )

    def predict(
        self,
        image_path: Path,
    ) -> dict[str, Any]:
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        image_tensor = (
            self.transform(image)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(
                output,
                dim=1,
            )[0]

        predicted_index = int(
            probabilities.argmax().item()
        )

        return {
            "image": str(image_path),
            "prediction": self.class_names[predicted_index],
            "confidence": float(
                probabilities[predicted_index].item()
            ),
            "real_probability": float(
                probabilities[0].item()
            ),
            "fake_probability": float(
                probabilities[1].item()
            ),
        }