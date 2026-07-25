from pathlib import Path
import random


class DeepfakeDetector:
    """
    Temporary AI detector.

    Later this class will load a trained deep learning model
    such as EfficientNet or Xception.
    """

    def __init__(self):
        self.model_loaded = True

    def predict(self, face_paths: list[str]) -> dict:

        if len(face_paths) == 0:
            return {
                "prediction": "No face detected",
                "confidence": 0.0
            }

        confidence = round(random.uniform(0.80, 0.99), 2)

        prediction = random.choice([
            "Likely Real",
            "Likely Fake"
        ])

        return {
            "prediction": prediction,
            "confidence": confidence,
            "faces_processed": len(face_paths)
        }