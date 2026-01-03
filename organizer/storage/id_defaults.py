"""Helpers for resolving default IDs across storage-backed operations."""

from __future__ import annotations

from sqlalchemy import func, select

from storage.index_models import Snapshot
from storage.manager import StorageManager
from storage.training_models import LabelRun
from storage.work_models import Run


def get_latest_run(storage_manager: StorageManager) -> Run | None:
    """Return the most recent run in work.db."""
    with storage_manager.get_work_session() as session:
        run = (
            session.execute(select(Run).order_by(Run.id.desc()).limit(1))
            .scalars()
            .first()
        )
        if run is not None:
            session.expunge(run)
        return run


def get_latest_run_for_snapshot(
    storage_manager: StorageManager, snapshot_id: int
) -> Run | None:
    """Return the most recent run for a given snapshot in work.db."""
    with storage_manager.get_work_session() as session:
        run = (
            session.execute(
                select(Run)
                .where(Run.snapshot_id == snapshot_id)
                .order_by(Run.id.desc())
                .limit(1)
            )
            .scalars()
            .first()
        )
        if run is not None:
            session.expunge(run)
        return run


def get_latest_snapshot_id(storage_manager: StorageManager) -> int | None:
    """Return the latest snapshot ID from index.db."""
    with storage_manager.get_index_session(read_only=True) as session:
        return session.execute(select(func.max(Snapshot.snapshot_id))).scalar()


def get_latest_label_run_id(storage_manager: StorageManager) -> int | None:
    """Return the latest label run ID from training.db."""
    with storage_manager.get_training_session() as session:
        return session.execute(select(func.max(LabelRun.id))).scalar()


def get_effective_snapshot_id(
    storage_manager: StorageManager, snapshot_id: int | None
) -> int:
    """Return provided snapshot_id, or default to the newest snapshot."""
    if snapshot_id is not None:
        return snapshot_id
    latest_snapshot_id = get_latest_snapshot_id(storage_manager)
    if latest_snapshot_id is None:
        raise ValueError(f"No snapshots found in {storage_manager.index_path}")
    return latest_snapshot_id


def get_effective_label_run_id(
    storage_manager: StorageManager, label_run_id: int | None
) -> int:
    """Return provided label_run_id, or default to the newest label run."""
    if label_run_id is not None:
        return label_run_id
    latest_label_run_id = get_latest_label_run_id(storage_manager)
    if latest_label_run_id is None:
        raise ValueError(f"No label runs found in {storage_manager.training_path}")
    return latest_label_run_id
