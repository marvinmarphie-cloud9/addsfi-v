import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(
        self,
        results_root: Path = Path("results/training"),
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_directory = results_root / f"run_{timestamp}"
        self.run_directory.mkdir(parents=True, exist_ok=True)

        self.history_path = self.run_directory / "history.csv"
        self.metrics_path = self.run_directory / "metrics.json"
        self.checkpoint_path = self.run_directory / "best_model.pt"

    def save_history(
        self,
        history: dict[str, list[float]],
    ) -> None:
        epoch_count = len(history["train_loss"])

        with self.history_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_accuracy",
                    "validation_loss",
                    "validation_accuracy",
                    "learning_rate",
                ]
            )

            for index in range(epoch_count):
                writer.writerow(
                    [
                        index + 1,
                        history["train_loss"][index],
                        history["train_accuracy"][index],
                        history["validation_loss"][index],
                        history["validation_accuracy"][index],
                        history["learning_rate"][index],
                    ]
                )

    def save_metrics(
        self,
        metrics: dict[str, Any],
    ) -> None:
        with self.metrics_path.open(
            "w",
            encoding="utf-8",
        ) as json_file:
            json.dump(
                metrics,
                json_file,
                indent=4,
            )