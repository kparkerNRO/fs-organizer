"""Training database manager functions.

Provides utilities for managing training samples, label runs, and model runs
in the training database.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from storage.index_models import Node, Snapshot
from storage.manager import NodeKind, StorageManager
from storage.training_models import (
    LabelRun,
    TrainingSample,
)
from utils.filename_processing import clean_filename
from utils.text_processing import normalize_string

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


def load_and_index_nodes(
    session: Session, snapshot_id: int
) -> Tuple[Dict[int, Node], Dict[int, str], Dict[Optional[int], List[int]]]:
    """Load all nodes for a snapshot and build relationship indexes."""
    rows = session.execute(
        select(Node).where(Node.snapshot_id == snapshot_id)
    ).scalars()
    nodes_by_id: Dict[int, Node] = {}
    processed_name_by_id: Dict[int, str] = {}
    children_by_parent: Dict[Optional[int], List[int]] = {}
    for r in rows:
        nodes_by_id[r.node_id] = r
        processed_name_by_id[r.node_id] = clean_filename(r.name)
        children_by_parent.setdefault(r.parent_node_id, []).append(r.node_id)
    return nodes_by_id, processed_name_by_id, children_by_parent


def precompute_descendant_extensions(
    nodes_by_id: Dict[int, Node], children_by_parent: Dict[Optional[int], List[int]]
) -> Dict[int, Set[str]]:
    """Precompute the set of all descendant file extensions for each node using dynamic programming."""
    depth_groups: Dict[int, List[int]] = {}
    max_depth = 0
    for node_id, n in nodes_by_id.items():
        d = int(n.depth)
        depth_groups.setdefault(d, []).append(node_id)
        max_depth = max(max_depth, d)

    descendant_exts: Dict[int, Set[str]] = {nid: set() for nid in nodes_by_id}
    for d in range(max_depth, -1, -1):
        for nid in depth_groups.get(d, []):
            node = nodes_by_id[nid]
            if node.kind == NodeKind.FILE and node.ext:
                descendant_exts[nid].add(normalize_string(node.ext).strip("."))
            for cid in children_by_parent.get(nid, []):
                descendant_exts[nid].update(descendant_exts[cid])
    return descendant_exts
