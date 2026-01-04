"""Training database manager functions.

Provides utilities for managing training samples, label runs, and model runs
in the training database.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session
from storage.index_models import Node
from storage.manager import NodeKind
from storage.training_models import (
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


def _load_and_index_nodes(
    session: Session, snapshot_id: int
) -> Tuple[Dict[int, Node], Dict[int, str], Dict[Optional[int], List[int]]]:
    """Load all nodes for a snapshot and build relationship indexes."""
    rows = session.execute(select(Node).where(Node.snapshot_id == snapshot_id)).scalars()
    nodes_by_id: Dict[int, Node] = {}
    processed_name_by_id: Dict[int, str] = {}
    children_by_parent: Dict[Optional[int], List[int]] = {}
    for r in rows:
        nodes_by_id[r.id] = r
        processed_name_by_id[r.id] = clean_filename(r.name)
        children_by_parent.setdefault(r.parent_node_id, []).append(r.id)
    return nodes_by_id, processed_name_by_id, children_by_parent


def _precompute_descendant_extensions(
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


class FeatureNodeCore(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    snapshot_id: int
    node: Node
    parent: Node | None
    grandparent: Node | None
    child_nodes: list[Node]
    sibling_nodes: list[Node]
    descendent_extentions: list[str]

    max_siblings: int
    max_descendents: int
    max_children: int

    @property
    def grandparent_name(self):
        return self.grandparent.name if self.grandparent else None

    @property
    def parent_name(self):
        return self.parent.name if self.parent else None

    @property
    def child_names(self):
        return [node.name for node in self.child_nodes]

    @property
    def sibling_names(self):
        return [node.name for node in self.sibling_nodes]


def extract_feature_nodes(
    index_session: Session,
    snapshot_id: int,
    nodes: list[Node],
    max_siblings: int,
    max_descendents: int,
    max_children: int,
):
    nodes_by_id, processed_name_by_id, children_by_parent = _load_and_index_nodes(
        index_session, snapshot_id
    )

    descendant_exts = _precompute_descendant_extensions(nodes_by_id, children_by_parent)

    feature_nodes: list[FeatureNodeCore] = []

    for node in nodes:
        is_zip_file = node.ext and node.ext.lower() in (".zip", "zip")
        if node.kind != NodeKind.DIR and not is_zip_file:
            continue

        parent = nodes_by_id.get(node.parent_node_id) if node.parent_node_id else None
        grandparent = (
            nodes_by_id.get(parent.parent_node_id) if parent and parent.parent_node_id else None
        )

        child_nodes = [
            nodes_by_id[cid] for cid in children_by_parent.get(node.id, []) if cid in nodes_by_id
        ][:max_children]

        if parent:
            sibling_nodes = [
                nodes_by_id[sid]
                for sid in children_by_parent.get(parent.id, [])
                if sid != node.id and sid in nodes_by_id and nodes_by_id[sid].kind == NodeKind.DIR
            ][:max_siblings]
        else:
            sibling_nodes = []

        extensions = sorted(descendant_exts.get(node.id, set()))[:max_descendents]
        core_node = FeatureNodeCore(
            snapshot_id=snapshot_id,
            node=node,
            parent=parent,
            grandparent=grandparent,
            child_nodes=child_nodes,
            sibling_nodes=sibling_nodes,
            descendent_extentions=extensions,
            max_siblings=max_siblings,
            max_descendents=max_descendents,
            max_children=max_children,
        )

        feature_nodes.append(core_node)

    return feature_nodes
