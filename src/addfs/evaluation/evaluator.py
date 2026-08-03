import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        validation_loader: DataLoader,
        output_directory: Path,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.validation_loader = validation_loader
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = self.model.to(self.device)

    def evaluate(self) -> dict[str, Any]:
        self.model.eval()

        true_labels: list[int] = []
        predicted_labels: list[int] = []
        fake_probabilities: list[float] = []

        with torch.no_grad():
            for images, labels in self.validation_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)

                true_labels.extend(labels.cpu().tolist())
                predicted_labels.extend(predictions.cpu().tolist())
                fake_probabilities.extend(
                    probabilities[:, 1].cpu().tolist()
                )

        metrics = {
            "accuracy": accuracy_score(
                true_labels,
                predicted_labels,
            ),
            "precision": precision_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            ),
            "recall": recall_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            ),
            "f1_score": f1_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            ),
            "roc_auc": roc_auc_score(
                true_labels,
                fake_probabilities,
            ),
            "validation_samples": len(true_labels),
        }

        self._save_metrics(metrics)
        self._save_classification_report(
            true_labels,
            predicted_labels,
        )
        self._save_confusion_matrix(
            true_labels,
            predicted_labels,
        )
        self._save_roc_curve(
            true_labels,
            fake_probabilities,
        )

        return metrics

    def _save_metrics(
        self,
        metrics: dict[str, Any],
    ) -> None:
        metrics_path = self.output_directory / "evaluation_metrics.json"

        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

    def _save_classification_report(
        self,
        true_labels: list[int],
        predicted_labels: list[int],
    ) -> None:
        report = classification_report(
            true_labels,
            predicted_labels,
            target_names=["Real", "Fake"],
            zero_division=0,
        )

        report_path = (
            self.output_directory / "classification_report.txt"
        )

        report_path.write_text(report, encoding="utf-8")

    def _save_confusion_matrix(
        self,
        true_labels: list[int],
        predicted_labels: list[int],
    ) -> None:
        matrix = confusion_matrix(
            true_labels,
            predicted_labels,
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Real", "Fake"],
        )

        display.plot(values_format="d")
        plt.title("Deepfake Detection Confusion Matrix")
        plt.tight_layout()
        plt.savefig(
            self.output_directory / "confusion_matrix.png",
            dpi=200,
        )
        plt.close()

    def _save_roc_curve(
        self,
        true_labels: list[int],
        fake_probabilities: list[float],
    ) -> None:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            true_labels,
            fake_probabilities,
        )

        auc_score = roc_auc_score(
            true_labels,
            fake_probabilities,
        )

        plt.figure()
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"ROC-AUC = {auc_score:.3f}",
        )
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Deepfake Detection ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            self.output_directory / "roc_curve.png",
            dpi=200,
        )
        plt.close()