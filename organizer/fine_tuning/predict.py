import csv
from pathlib import Path
from typing import List

from .models.base import BaseClassifier
from storage.training_models import TrainingSample


def predict(classifier: BaseClassifier, samples: List[TrainingSample]):
    """Run predictions using the given classifier."""
    return classifier.predict(samples)


def save_predictions_to_csv(
    output_file: Path,
    samples: List[TrainingSample],
    predictions: List[str],
    confidences: List[float],
):
    """Save predictions to a CSV file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "name",
                "true_label",
                "predicted_label",
                "confidence",
                "is_correct",
            ]
        )

        for sample, pred, conf in zip(samples, predictions, confidences):
            is_correct = ""
            if sample.label:
                is_correct = "1" if sample.label == pred else "0"

            writer.writerow(
                [
                    sample.sample_id,
                    sample.name_raw,
                    sample.label or "",
                    pred,
                    f"{conf:.4f}",
                    is_correct,
                ]
            )
