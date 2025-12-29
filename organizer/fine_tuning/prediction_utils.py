"""Shared helpers for prediction workflows."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Callable

from sqlalchemy.orm import Session

from fine_tuning.prediction_db import create_model_run, get_newest_label_run_id
from storage.training_models import ModelRun, TrainingSample

LogFn = Callable[[str], None]


def resolve_label_run_id(
    session: Session,
    label_run_id: int | None,
    *,
    log: LogFn,
) -> int:
    if label_run_id is None:
        effective_label_run_id = get_newest_label_run_id(session)
        if effective_label_run_id is None:
            log("Error: No label runs found in database")
            raise ValueError("No label runs found in database")
        log(f"Using newest label run: {effective_label_run_id}")
        return effective_label_run_id

    log(f"Using specified label run: {label_run_id}")
    return label_run_id


def save_predictions_csv(
    output_file: Path,
    samples: list[TrainingSample],
    predictions: list[str],
    confidences: list[float],
) -> None:
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


def record_model_run(
    session: Session,
    *,
    model_path: Path | None,
    taxonomy: str,
    use_baseline: bool,
    config: dict,
    metrics: dict,
    run_type: str,
    sample_count: int,
) -> ModelRun:
    run = create_model_run(
        session,
        model_path=str(model_path) if model_path else None,
        taxonomy=taxonomy,
        use_baseline=use_baseline,
        config=config,
        run_type=run_type,
    )

    run.status = "completed"
    run.finished_at = datetime.now().isoformat()
    run.test_samples_count = sample_count

    if metrics:
        run.final_val_accuracy = metrics.get("accuracy")
        run.final_val_f1 = metrics.get("macro_f1")

    metrics_summary = ""
    if metrics:
        metrics_summary = (
            f", Accuracy: {metrics.get('accuracy', 0):.4f}, "
            f"Macro-F1: {metrics.get('macro_f1', 0):.4f}, "
            f"Weighted-F1: {metrics.get('weighted_f1', 0):.4f}"
        )

    split_label = config.get("split") or "all"
    run.notes = (
        f"Run type: {run_type}, Taxonomy: {taxonomy}, Split: {split_label}{metrics_summary}"
    )

    session.commit()
    return run
