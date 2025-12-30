import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fine_tuning.classifiers.ml_classifiers import SetFitClassifier, ZeroShotClassifier
from fine_tuning.services.evaluation import evaluate_predictions
from fine_tuning.settings import CommonSettings
from fine_tuning.services.common import (
    get_effective_label_run_id,
    load_samples,
)
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from storage.manager import StorageManager
from storage.training_models import ModelRun, SamplePrediction, TrainingSample

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


class FullPredictSettings(CommonSettings, PredictSettings):
    pass


class FullZeroShotSettings(CommonSettings, ZeroShotSettings):
    pass


def save_predictions_to_db(
    session: Session,
    samples: List[TrainingSample],
    predictions: List[str],
    confidences: List[float],
    probabilities: List[Union[List[float], Dict[str, float]]],
    run_id: int,
    prediction_type: str = "test",
) -> int:
    """Save predictions to database.

    Args:
        session: SQLAlchemy session
        samples: List of samples
        predictions: List of predicted labels
        confidences: List of confidence scores
        probabilities: List of probability dicts or lists
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

        # Convert probabilities list to dict if needed
        if isinstance(probs, list):
            probs_dict = {f"label_{i}": p for i, p in enumerate(probs)}
        else:
            probs_dict = probs

        prediction_obj = SamplePrediction(
            run_id=run_id,
            sample_id=sample.sample_id,
            predicted_label=pred,
            confidence=conf,
            probabilities_json=json.dumps(probs_dict),
            true_label=sample.label,
            is_correct=is_correct,
            prediction_type=prediction_type,
        )
        prediction_objects.append(prediction_obj)

    session.add_all(prediction_objects)
    session.commit()

    return len(prediction_objects)


def create_model_run(
    session: Session,
    model_path: str | None,
    taxonomy: str,
    use_baseline: bool,
    config: Dict,
    run_type: str | None = None,
    training_data_source: str | None = None,
) -> ModelRun:
    """Create a new model run record.

    Args:
        session: SQLAlchemy session
        model_path: Path to fine-tuned model (or None if baseline)
        taxonomy: Taxonomy version
        use_baseline: Whether using baseline model
        config: Configuration dict
        run_type: Type of run ('training', 'evaluation', 'baseline'). Auto-detected if None.
        training_data_source: Description of training data source (optional)

    Returns:
        ModelRun object
    """
    # Auto-detect run type if not specified
    if run_type is None:
        run_type = "baseline" if use_baseline else "evaluation"

    # Store run type and taxonomy clearly in base_model_id
    base_model_id = f"setfit-{run_type}-{taxonomy}"

    # Store model path or baseline indicator in model_version
    if use_baseline:
        model_version = f"baseline-{taxonomy}"
    else:
        model_version = model_path if model_path else f"unknown-{taxonomy}"

    run = ModelRun(
        started_at=datetime.now().isoformat(),
        status="running",
        run_type=run_type,
        base_model_id=base_model_id,
        model_version=model_version,
        model_type="setfit",
        taxonomy=taxonomy,
        training_data_source=training_data_source,
        hyperparameters_json=json.dumps(config),
    )

    session.add(run)
    session.commit()
    session.refresh(run)

    return run


def create_and_save_run_results(
    session: Session,
    settings: Union[FullPredictSettings, FullZeroShotSettings],
    classifier: Union[SetFitClassifier, ZeroShotClassifier],
    samples: List[TrainingSample],
    predictions: List[str],
    confidences: List[float],
    probabilities: List[Union[List[float], Dict[str, float]]],
    metrics: Dict[str, Any],
) -> None:
    """Creates a ModelRun and saves all results to the database."""
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

    # Always save predictions to database
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
