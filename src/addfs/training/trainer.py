from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)

from .experiment import ExperimentLogger


class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        checkpoint_path: Path | None = None,
        learning_rate: float = 0.001,
        use_pretrained_weights: bool = True,
        patience: int = 3,
        scheduler_patience: int = 1,
    ) -> None:
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.experiment_logger = ExperimentLogger()

        self.checkpoint_path = (
            checkpoint_path
            if checkpoint_path is not None
            else self.experiment_logger.checkpoint_path
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if use_pretrained_weights
            else None
        )

        self.model = efficientnet_b0(weights=weights)

        input_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(
            input_features,
            2,
        )

        self.model = self.model.to(self.device)

        self.loss_function = nn.CrossEntropyLoss()

        self.optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=scheduler_patience,
        )

        self.best_validation_accuracy = 0.0
        self.early_stopping_patience = patience
        self.epochs_without_improvement = 0

    def train(
        self,
        epochs: int = 5,
    ) -> dict[str, list[float]]:
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "validation_loss": [],
            "validation_accuracy": [],
            "learning_rate": [],
        }

        print(f"Training device: {self.device}")

        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = self._train_one_epoch()

            validation_loss, validation_accuracy = (
                self._validate_one_epoch()
            )

            self.scheduler.step(validation_loss)

            current_learning_rate = (
                self.optimizer.param_groups[0]["lr"]
            )

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["validation_loss"].append(validation_loss)
            history["validation_accuracy"].append(
                validation_accuracy
            )
            history["learning_rate"].append(
                current_learning_rate
            )

            print(
                f"\nEpoch {epoch}/{epochs}"
                f"\nTrain loss: {train_loss:.4f}"
                f" | Train accuracy: {train_accuracy:.2%}"
                f"\nValidation loss: {validation_loss:.4f}"
                f" | Validation accuracy: "
                f"{validation_accuracy:.2%}"
                f"\nLearning rate: "
                f"{current_learning_rate:.8f}"
            )

            if validation_accuracy > self.best_validation_accuracy:
                self.best_validation_accuracy = (
                    validation_accuracy
                )

                self.epochs_without_improvement = 0

                self._save_checkpoint(
                    epoch=epoch,
                    validation_accuracy=validation_accuracy,
                )

                print(
                    "Saved new best checkpoint: "
                    f"{self.checkpoint_path}"
                )
            else:
                self.epochs_without_improvement += 1

                print(
                    "No improvement for "
                    f"{self.epochs_without_improvement} epoch(s)."
                )

                if (
                    self.epochs_without_improvement
                    >= self.early_stopping_patience
                ):
                    print("Early stopping triggered.")
                    break

        self.experiment_logger.save_history(history)

        self.experiment_logger.save_metrics(
            {
                "device": str(self.device),
                "requested_epochs": epochs,
                "completed_epochs": len(
                    history["train_loss"]
                ),
                "best_validation_accuracy": (
                    self.best_validation_accuracy
                ),
                "checkpoint_path": str(
                    self.checkpoint_path
                ),
                "training_samples": len(
                    self.train_loader.dataset
                ),
                "validation_samples": len(
                    self.validation_loader.dataset
                ),
                "batch_size": self.train_loader.batch_size,
                "model_name": "efficientnet_b0",
                "early_stopping_patience": (
                    self.early_stopping_patience
                ),
                "final_learning_rate": (
                    self.optimizer.param_groups[0]["lr"]
                ),
            }
        )

        print(
            "\nExperiment results saved to: "
            f"{self.experiment_logger.run_directory}"
        )

        return history

    def _train_one_epoch(
        self,
    ) -> tuple[float, float]:
        self.model.train()

        running_loss = 0.0
        correct_predictions = 0
        sample_count = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.loss_function(
                outputs,
                labels,
            )

            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size
            sample_count += batch_size

            predictions = outputs.argmax(dim=1)

            correct_predictions += (
                predictions == labels
            ).sum().item()

        average_loss = running_loss / sample_count
        accuracy = correct_predictions / sample_count

        return average_loss, accuracy

    def _validate_one_epoch(
        self,
    ) -> tuple[float, float]:
        self.model.eval()

        running_loss = 0.0
        correct_predictions = 0
        sample_count = 0

        with torch.no_grad():
            for images, labels in self.validation_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                loss = self.loss_function(
                    outputs,
                    labels,
                )

                batch_size = labels.size(0)

                running_loss += loss.item() * batch_size
                sample_count += batch_size

                predictions = outputs.argmax(dim=1)

                correct_predictions += (
                    predictions == labels
                ).sum().item()

        average_loss = running_loss / sample_count
        accuracy = correct_predictions / sample_count

        return average_loss, accuracy

    def _save_checkpoint(
        self,
        epoch: int,
        validation_accuracy: float,
    ) -> None:
        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),
            "validation_accuracy": validation_accuracy,
            "class_names": ["0_real", "1_fake"],
            "model_name": "efficientnet_b0",
        }

        torch.save(
            checkpoint,
            self.checkpoint_path,
        )