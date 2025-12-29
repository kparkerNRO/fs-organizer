"""
Evaluation utilities for the fine-tuning pipeline.
"""
from collections import defaultdict
from typing import Dict, List, Set

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


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
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=sorted(labels), zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=sorted(labels), zero_division=0)

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
                zero_division=0,
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
