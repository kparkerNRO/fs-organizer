"""Training database manager functions.

Provides utilities for managing training samples, label runs, and model runs
in the training database.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from storage.index_models import Snapshot
from storage.manager import StorageManager
from storage.training_models import (
    LabelRun,
    ModelRun,
    SamplePrediction,
    TrainingSample,
)

logger = logging.getLogger(__name__)


def load_samples(
    session: Session,
    split: str | None = None,
    labeled_only: bool = False,
    label_run_id: int | None = None,
) -> List[TrainingSample]:
    """Load training samples from database.

    Args:
        session: SQLAlchemy session
        split: Optional split filter ('train', 'validation', 'test')
        labeled_only: Only load samples with labels
        label_run_id: Optional label run ID to filter by

    Returns:
        List of TrainingSample objects
    """
    query = select(TrainingSample)

    if label_run_id is not None:
        query = query.where(TrainingSample.label_run_id == label_run_id)

    if split:
        query = query.where(TrainingSample.split == split)

    if labeled_only:
        query = query.where(TrainingSample.label.isnot(None))
        query = query.where(TrainingSample.label != "")

    samples = session.execute(query).scalars().all()
    return list(samples)


def get_newest_label_run_id(session: Session) -> int | None:
    """Get the newest (highest ID) label run from the database.

    Args:
        session: SQLAlchemy session

    Returns:
        The ID of the newest label run, or None if no label runs exist
    """
    result = session.execute(
        select(LabelRun.id).order_by(LabelRun.id.desc()).limit(1)
    ).scalar()
    return result


def save_predictions_to_db(
    session: Session,
    samples: List[TrainingSample],
    predictions: List[str],
    confidences: List[float],
    probabilities: List[List[float]],
    run_id: int,
    prediction_type: str = "test",
) -> int:
    """Save predictions to database.

    Args:
        session: SQLAlchemy session
        samples: List of samples
        predictions: List of predicted labels
        confidences: List of confidence scores
        probabilities: List of probability lists
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


def get_highest_snapshot_id(manager: StorageManager) -> int:
    """Get the highest snapshot_id from the index database."""
    with manager.get_index_session(read_only=True) as session:
        result = session.execute(select(func.max(Snapshot.snapshot_id))).scalar()
        if result is None:
            raise ValueError(f"No snapshots found in {manager.get_index_db_path()}")
        return result


def get_effective_label_run_id(session: Session, label_run_id: int | None) -> int:
    """Get the effective label run ID, defaulting to the newest if not specified."""
    if label_run_id is not None:
        logger.info(f"Using specified label run: {label_run_id}")
        return label_run_id

    effective_label_run_id = get_newest_label_run_id(session)
    if effective_label_run_id is None:
        raise ValueError("No label runs found in database")
    logger.info(f"Using newest label run: {effective_label_run_id}")
    return effective_label_run_id
