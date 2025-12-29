"""
Shared utility functions for the fine-tuning module.
"""

from typing import Dict, List, Optional, Set, Tuple
from sqlalchemy import select
from sqlalchemy.orm import Session
from storage.index_models import Node
from storage.manager import NodeKind
from utils.filename_processing import clean_filename
from utils.text_processing import normalize_string


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
