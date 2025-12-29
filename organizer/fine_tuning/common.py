import json
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session
from storage.index_models import Snapshot
from storage.training_models import LabelRun, ModelRun, TrainingSample


def get_db_session(db_path: str) -> Session:
    """Get a new SQLAlchemy session."""
    engine = create_engine(f"sqlite:///{db_path}")
    return Session(engine)


def load_samples(
    session: Session,
    split: Optional[str] = None,
    labeled_only: bool = False,
    label_run_id: Optional[int] = None,
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


def get_newest_label_run_id(session: Session) -> Optional[int]:
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


def get_highest_snapshot_id(session: Session) -> Optional[int]:
    """Get the highest snapshot_id from the index database.

    Args:
        session: SQLAlchemy session for the index database

    Returns:
        The highest snapshot_id, or None if no snapshots are found.
    """
    result = session.execute(select(func.max(Snapshot.snapshot_id))).scalar()
    return result


def create_model_run(
    session: Session,
    model_path: Optional[str],
    taxonomy: str,
    use_baseline: bool,
    config: Dict,
    run_type: Optional[str] = None,
    training_data_source: Optional[str] = None,
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
