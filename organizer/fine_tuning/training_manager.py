"""Training database data access layer.

Provides utilities for querying and modifying the training.db database.
Session management is handled by the StorageManager.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from fine_tuning.training_models import (
    LabelRun,
    ModelRun,
    SamplePrediction,
    TrainingSample,
)


def load_samples(
    session: Session,
    split: Optional[str] = None,
    labeled_only: bool = False,
    label_run_id: Optional[int] = None,
) -> List[TrainingSample]:
    """Load training samples from database."""
    query = select(TrainingSample)
    if label_run_id is not None:
        query = query.where(TrainingSample.label_run_id == label_run_id)
    if split:
        query = query.where(TrainingSample.split == split)
    if labeled_only:
        query = query.where(
            TrainingSample.label.isnot(None) & (TrainingSample.label != "")
        )
    return list(session.execute(query).scalars().all())


def get_newest_label_run_id(session: Session) -> Optional[int]:
    """Get the newest (highest ID) label run from the database."""
    return session.execute(
        select(LabelRun.id).order_by(LabelRun.id.desc()).limit(1)
    ).scalar_one_or_none()


def save_predictions_to_db(
    session: Session,
    samples: List[TrainingSample],
    predictions: List[str],
    confidences: List[float],
    probabilities: List[Dict[str, float]],
    run_id: int,
    prediction_type: str = "test",
) -> int:
    """Save predictions to database."""
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


def create_model_run(
    session: Session,
    model_path: Optional[str],
    taxonomy: str,
    use_baseline: bool,
    config: Dict,
    run_type: Optional[str] = None,
    training_data_source: Optional[str] = None,
) -> ModelRun:
    """Create a new model run record."""
    if run_type is None:
        run_type = "baseline" if use_baseline else "evaluation"

    base_model_id = f"setfit-{run_type}-{taxonomy}"
    model_version = (
        f"baseline-{taxonomy}"
        if use_baseline
        else model_path
        if model_path
        else f"unknown-{taxonomy}"
    )

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
