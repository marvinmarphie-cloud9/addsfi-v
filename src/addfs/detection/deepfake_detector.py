from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import Compose


class DeepfakeDetector:
    """
    EfficientNet-B0 binary deepfake classifier.

    The model architecture is real, but meaningful predictions require
    trained deepfake-detection weights at models/trained/deepfake_model.pt.
    """

    def __init__(
        self,
        model_path: str | Path = "models/trained/deepfake_model.pt",
    ) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        weights = EfficientNet_B0_Weights.DEFAULT
        self.transform: Compose = weights.transforms()

        self.model = efficientnet_b0(weights=weights)

        input_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(input_features, 2)

        self.model_path = Path(model_path)
        self.model_ready = self.model_path.exists()

        if self.model_ready:
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, face_paths: list[str]) -> dict:
        if not face_paths:
            return {
                "prediction": "No face detected",
                "confidence": 0.0,
                "faces_processed": 0,
                "model_ready": self.model_ready,
            }

        if not self.model_ready:
            return {
                "prediction": "Model not trained",
                "confidence": 0.0,
                "faces_processed": len(face_paths),
                "model_ready": False,
            }

        fake_probabilities: list[float] = []

        with torch.inference_mode():
            for face_path in face_paths:
                image = Image.open(face_path).convert("RGB")
                tensor = self.transform(image).unsqueeze(0).to(self.device)

                logits = self.model(tensor)
                probabilities = torch.softmax(logits, dim=1)

                fake_probability = float(probabilities[0, 1].item())
                fake_probabilities.append(fake_probability)

        average_fake_probability = sum(fake_probabilities) / len(
            fake_probabilities
        )

        prediction = (
            "Likely Fake"
            if average_fake_probability >= 0.5
            else "Likely Real"
        )

        confidence = (
            average_fake_probability
            if prediction == "Likely Fake"
            else 1.0 - average_fake_probability
        )

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "fake_probability": round(average_fake_probability, 4),
            "faces_processed": len(face_paths),
            "model_ready": True,
            "device": str(self.device),
        }