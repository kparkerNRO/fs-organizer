import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fine_tuning.classifiers import SetFitClassifier, ZeroShotClassifier
from fine_tuning.services.evaluation import evaluate_predictions
from fine_tuning.settings import CommonSettings
from fine_tuning.training_manager import (
    create_model_run,
    get_effective_label_run_id,
    load_samples,
    save_predictions_to_db,
)
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from storage.manager import StorageManager
from storage.training_models import TrainingSample

logger = logging.getLogger(__name__)


class PredictSettings(BaseModel):
    """Settings for the 'predict' command."""

    model_path: Optional[Path] = Field(
        None,
        description="Path to fine-tuned SetFit model (required unless --use-baseline)",
    )
    use_baseline: bool = Field(
        False,
        description="Use baseline pre-trained model without fine-tuning",
    )
    split: Optional[str] = Field(
        None,
        description="Only evaluate on specific split: train, validation, or test",
    )
    labeled_only: bool = Field(
        False,
        description="Only run on samples with labels (for evaluation)",
    )
    save_predictions: bool = Field(
        False,
        description="Save predictions to database",
    )
    output_file: Optional[Path] = Field(
        None,
        description="Save predictions to CSV file",
    )


class ZeroShotSettings(BaseModel):
    """Settings for the 'zero-shot' command."""

    split: Optional[str] = Field(
        None,
        description="Only evaluate on specific split: train, validation, or test",
    )
    labeled_only: bool = Field(
        False,
        description="Only run on samples with labels (for evaluation)",
    )
    save_predictions: bool = Field(
        False,
        description="Save predictions to database",
    )
    output_file: Optional[Path] = Field(
        None,
        description="Save predictions to CSV file",
    )


class FullPredictSettings(CommonSettings, PredictSettings):
    pass


class FullZeroShotSettings(CommonSettings, ZeroShotSettings):
    pass


def save_predictions_to_csv(
    output_file: Path,
    samples: List[TrainingSample],
    predictions: List[str],
    confidences: List[float],
) -> None:
    """Save predictions to a CSV file."""
    logger.info(f"Saving predictions to {output_file}...")
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
            is_correct = "1" if sample.label and sample.label == pred else "0"
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
    logger.info(f"Saved {len(samples)} predictions to {output_file}")


def create_and_save_run_results(
    session: Session,
    settings: Union[FullPredictSettings, FullZeroShotSettings],
    classifier: Union[SetFitClassifier, ZeroShotClassifier],
    samples: List[TrainingSample],
    predictions: List[str],
    confidences: List[float],
    probabilities: List[List[float]],
    metrics: Dict[str, Any],
) -> None:
    """Creates a ModelRun, saves all results to the database and optionally to CSV."""
    run_type_label = (
        "zero-shot" if isinstance(classifier, ZeroShotClassifier) else "fine-tuned"
    )
    if isinstance(settings, FullPredictSettings) and settings.use_baseline:
        run_type_label = "baseline"

    logger.info("Saving results to database...")
    config = settings.model_dump()
    if metrics:
        config["metrics"] = metrics

    model_path = getattr(settings, "model_path", None)
    run = create_model_run(
        session,
        model_path=str(model_path) if model_path else None,
        taxonomy=settings.taxonomy,
        use_baseline=getattr(settings, "use_baseline", False),
        config=config,
        run_type=run_type_label,
    )

    run.status = "completed"
    run.finished_at = datetime.now().isoformat()
    run.test_samples_count = len(samples)

    if metrics:
        run.final_val_accuracy = metrics.get("accuracy")
        run.final_val_f1 = metrics.get("macro_f1")

    if metrics:
        metrics_summary = (
            f", Accuracy: {metrics.get('accuracy', 0):.4f}, "
            f"Macro-F1: {metrics.get('macro_f1', 0):.4f}"
        )
    else:
        metrics_summary = ""
    run.notes = (
        f"Run type: {run_type_label}, Taxonomy: {settings.taxonomy}, "
        f"Split: {settings.split or 'all'}{metrics_summary}"
    )

    session.commit()
    logger.info(f"Saved run metadata (run_id={run.run_id})")

    if settings.save_predictions:
        num_saved = save_predictions_to_db(
            session,
            samples,
            predictions,
            confidences,
            probabilities,
            run_id=run.run_id,
            prediction_type=settings.split or "all",
        )
        logger.info(f"Saved {num_saved} predictions to database")

    if settings.output_file:
        save_predictions_to_csv(settings.output_file, samples, predictions, confidences)


def predict_and_evaluate(
    settings: Union[FullPredictSettings, FullZeroShotSettings],
    classifier: Union[SetFitClassifier, ZeroShotClassifier],
    manager: StorageManager,
) -> None:
    """Shared logic for prediction, evaluation, and saving results."""
    with manager.get_training_session() as session:
        effective_label_run_id = get_effective_label_run_id(
            session, settings.label_run_id
        )

        samples = load_samples(
            session,
            split=settings.split,
            labeled_only=settings.labeled_only,
            label_run_id=effective_label_run_id,
        )
        logger.info(
            f"Loaded {len(samples)} samples from {manager.get_training_db_path()}"
        )

        if not samples:
            logger.info("No samples found.")
            return

        logger.info("Running predictions...")
        predictions, confidences, probabilities = classifier.predict(samples)

        y_true = [s.label for s in samples if s.label]
        y_pred_labeled = [
            pred for i, pred in enumerate(predictions) if samples[i].label
        ]

        metrics = {}
        if y_true and y_pred_labeled:
            logger.info(f"Evaluating {len(y_true)} labeled samples...")
            metrics = evaluate_predictions(
                y_true, y_pred_labeled, classifier.labels, verbose=True
            )
        else:
            logger.info("No labeled samples found. Skipping evaluation.")

        create_and_save_run_results(
            session,
            settings,
            classifier,
            samples,
            predictions,
            confidences,
            probabilities,
            metrics,
        )


def predict_setfit(settings: FullPredictSettings, manager: StorageManager) -> None:
    if settings.use_baseline:
        logger.info(
            f"Initializing baseline SetFit model (taxonomy={settings.taxonomy})..."
        )
        classifier = SetFitClassifier(taxonomy=settings.taxonomy, use_baseline=True)
    else:
        logger.info(f"Loading fine-tuned model from {settings.model_path}...")
        classifier = SetFitClassifier(
            model_path=str(settings.model_path), taxonomy=settings.taxonomy
        )

    predict_and_evaluate(settings, classifier, manager)


def predict_zero_shot(settings: FullZeroShotSettings, manager: StorageManager) -> None:
    logger.info(f"Initializing zero-shot classifier (taxonomy={settings.taxonomy})...")
    classifier = ZeroShotClassifier(taxonomy=settings.taxonomy)
    predict_and_evaluate(settings, classifier, manager)
