import json
from collections import defaultdict
from typing import Dict, List, Set

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sqlalchemy.orm import Session

from storage.training_models import SamplePrediction, TrainingSample


def evaluate_predictions(
    y_true: List[str],
    y_pred: List[str],
    labels: Set[str],
    verbose: bool = True,
) -> Dict:
    """Evaluate predictions and print metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Set of valid labels
        verbose: Print detailed metrics

    Returns:
        Dictionary of metrics
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=sorted(labels))
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=sorted(labels))

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_samples": len(y_true),
    }

    if verbose:
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"\nTotal Samples: {len(y_true)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1-Score: {macro_f1:.4f}")
        print(f"Weighted F1-Score: {weighted_f1:.4f}")

        print("\n" + "-" * 80)
        print("CLASSIFICATION REPORT")
        print("-" * 80)
        print(
            classification_report(
                y_true,
                y_pred,
                labels=sorted(labels),
                target_names=sorted(labels),
                digits=4,
            )
        )

        print("\n" + "-" * 80)
        print("CONFUSION MATRIX")
        print("-" * 80)
        cm = confusion_matrix(y_true, y_pred, labels=sorted(labels))
        label_list = sorted(labels)

        # Print header
        print(f"{'':20s}", end="")
        for label in label_list:
            print(f"{label[:15]:>15s}", end=" ")
        print()

        # Print rows
        for i, true_label in enumerate(label_list):
            print(f"{true_label[:20]:20s}", end="")
            for j in range(len(label_list)):
                print(f"{cm[i, j]:>15d}", end=" ")
            print()

        # Print most common errors
        print("\n" + "-" * 80)
        print("MOST COMMON ERRORS")
        print("-" * 80)

        errors = defaultdict(int)
        for true, pred in zip(y_true, y_pred):
            if true != pred:
                errors[(true, pred)] += 1

        sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)
        for (true, pred), count in sorted_errors[:10]:
            print(f"  {true:20s} -> {pred:20s}: {count:4d} errors")

    return metrics


def save_predictions_to_db(
    session: Session,
    samples: List[TrainingSample],
    predictions: List[str],
    confidences: List[float],
    probabilities: List[Dict[str, float]],
    run_id: int,
    prediction_type: str = "test",
) -> int:
    """Save predictions to database.

    Args:
        session: SQLAlchemy session
        samples: List of samples
        predictions: List of predicted labels
        confidences: List of confidence scores
        probabilities: List of probability dicts
        run_id: TrainingRun ID
        prediction_type: Type of prediction ('train', 'validation', 'test')

    Returns:
        Number of predictions saved
    """
    prediction_objects = []

    for sample, pred, conf, probs in zip(
        samples, predictions, confidences, probabilities
    ):
        is_correct = None
        if sample.label:
            is_correct = sample.label == pred

        prediction_obj = SamplePrediction(
            run_id=run_id,
            sample_id=sample.sample_id,
            predicted_label=pred,
            confidence=conf,
            probabilities_json=json.dumps(probs),
            true_label=sample.label,
            is_correct=is_correct,
            prediction_type=prediction_type,
        )
        prediction_objects.append(prediction_obj)

    session.add_all(prediction_objects)
    session.commit()

    return len(prediction_objects)
