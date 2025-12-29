"""Extract features for model training."""

import json
from typing import Dict, List, Optional, Set

from sqlalchemy import select
from sqlalchemy.orm import Session
from storage.index_models import Node
from storage.manager import NodeKind
from storage.training_models import LabelRun, TrainingSample
from utils.config import Config
from utils.filename_processing import clean_filename
from utils.text_processing import has_matching_token, normalize_string, tokenize_string

from .heuristic_classifier import COLLAB_MARKERS


def _load_and_index_nodes(
    session: Session, snapshot_id: int
) -> Tuple[Dict[int, Node], Dict[int, str], Dict[Optional[int], List[int]]]:
    """Load nodes from DB and build indexes."""
    rows = session.execute(select(Node).where(Node.snapshot_id == snapshot_id)).scalars()
    nodes_by_id: Dict[int, Node] = {}
    processed_name_by_id: dict[int, str] = {}
    children_by_parent: Dict[Optional[int], List[int]] = {}
    for r in rows:
        nodes_by_id[r.node_id] = r
        processed_name_by_id[r.node_id] = clean_filename(r.name)
        children_by_parent.setdefault(r.parent_node_id, []).append(r.node_id)
    return nodes_by_id, processed_name_by_id, children_by_parent


def _precompute_descendant_extensions(
    nodes_by_id: Dict[int, Node], children_by_parent: Dict[Optional[int], List[int]]
) -> Dict[int, Set[str]]:
    """Precompute descendant file extensions for each node."""
    depth_groups: Dict[int, List[int]] = {}
    max_depth = 0
    for node_id, n in nodes_by_id.items():
        d = int(n.depth)
        depth_groups.setdefault(d, []).append(node_id)
        if d > max_depth:
            max_depth = d

    descendant_exts: Dict[int, Set[str]] = {nid: set() for nid in nodes_by_id.keys()}
    for d in range(max_depth, -1, -1):
        for nid in depth_groups.get(d, []):
            n = nodes_by_id[nid]
            if n.kind == NodeKind.FILE and n.ext:
                descendant_exts[nid].add(normalize_string(n.ext).strip("."))
            for cid in children_by_parent.get(nid, []):
                descendant_exts[nid].update(descendant_exts[cid])
    return descendant_exts


def _build_feature_text(
    node: Node,
    name_norm: str,
    parent: Optional[Node],
    grandparent: Optional[Node],
    sibling_names: List[str],
    child_names: List[str],
    exts: List[str],
    flags: Dict[str, bool],
) -> str:
    """Build the text feature for a single node."""
    return " | ".join(
        [
            f"gp:{grandparent.name if grandparent else ''}",
            f"p:{parent.name if parent else ''}",
            f"t:{name_norm}",
            f"depth:{node.depth}",
            "sibs:" + " ".join(sibling_names),
            "children:" + " ".join(child_names),
            "exts:" + " ".join(exts),
            f"flags:"
            + " ".join(f"{k}={int(v)}" for k, v in flags.items()),
        ]
    )


def _create_training_sample(
    node: Node,
    name_norm: str,
    parent: Optional[Node],
    grandparent: Optional[Node],
    child_names: List[str],
    sibling_names: List[str],
    exts: List[str],
    flags: Dict[str, bool],
    text: str,
    label_run: LabelRun,
) -> TrainingSample:
    """Create a TrainingSample object."""
    return TrainingSample(
        snapshot_id=node.snapshot_id,
        node_id=node.node_id,
        name_raw=node.name,
        name_norm=name_norm,
        parent_name_norm=parent.name if parent else None,
        grandparent_name_norm=grandparent.name if grandparent else None,
        kind=node.kind,
        file_source=node.file_source,
        depth=int(node.depth),
        child_names_topk_json=json.dumps(child_names),
        sibling_names_topk_json=json.dumps(sibling_names),
        descendant_file_exts_topk_json=json.dumps(exts),
        has_collab_cue=flags.get("has_collab_cue", False),
        looks_like_format=flags.get("looks_like_format", False),
        child_has_media_type_cue=flags.get("child_has_media_type_cue", False),
        child_has_variant_hint=flags.get("child_has_variant_hint", False),
        child_has_format_cue=flags.get("child_has_format_cue", False),
        sibling_has_variant_hint=flags.get("sibling_has_variant_hint", False),
        text=text,
        label_run=label_run,
    )


def extract_features(
    index_session: Session,
    training_session: Session,
    snapshot_id: int,
    config: Config,
    label_run: LabelRun,
    *,
    child_cap: int = 20,
    sibling_cap: int = 20,
    ext_cap: int = 12,
    batch_size: int = 1000,
) -> int:
    """Extract features for all nodes in a snapshot and save to training database."""
    nodes_by_id, processed_name_by_id, children_by_parent = _load_and_index_nodes(
        index_session, snapshot_id
    )
    descendant_exts = _precompute_descendant_extensions(
        nodes_by_id, children_by_parent
    )

    samples: List[TrainingSample] = []
    total_saved = 0

    for nid, n in nodes_by_id.items():
        is_zip_file = n.ext and n.ext.lower() in (".zip", "zip")
        if n.kind != NodeKind.DIR and not is_zip_file:
            continue

        name_norm = processed_name_by_id[nid]
        parent_id = n.parent_node_id
        parent = nodes_by_id.get(parent_id) if parent_id is not None else None
        grandparent = (
            nodes_by_id.get(parent.parent_node_id)
            if parent and parent.parent_node_id is not None
            else None
        )

        child_ids = children_by_parent.get(nid, [])
        sibling_ids = (
            [sid for sid in children_by_parent.get(parent_id, []) if sid != nid]
            if parent_id in children_by_parent
            else []
        )

        child_names = sorted({processed_name_by_id[c] for c in child_ids})[:child_cap]
        sibling_names = sorted({processed_name_by_id[s] for s in sibling_ids})[:sibling_cap]

        child_token_bag = [t for cn in child_names for t in tokenize_string(cn)]
        sibling_token_bag = [t for sn in sibling_names for t in tokenize_string(sn)]

        flags = {
            "has_collab_cue": has_matching_token(
                tokenize_string(n.name), COLLAB_MARKERS
            )
            or (
                parent is not None
                and has_matching_token(tokenize_string(parent.name), COLLAB_MARKERS)
            ),
            "looks_like_format": config.is_media_type(n.name),
            "child_has_media_type_cue": has_matching_token(
                child_token_bag, config.media_types
            ),
            "child_has_variant_hint": has_matching_token(
                child_token_bag, config.variant_types
            ),
            "child_has_format_cue": any(
                cn.strip(".").lower() in config.format_types for cn in child_names
            ),
            "sibling_has_variant_hint": has_matching_token(
                sibling_token_bag, config.variant_types
            ),
        }

        exts = sorted(descendant_exts[nid])[:ext_cap]

        text = _build_feature_text(
            n, name_norm, parent, grandparent, sibling_names, child_names, exts, flags
        )
        sample = _create_training_sample(
            n, name_norm, parent, grandparent, child_names, sibling_names, exts, flags, text, label_run
        )
        samples.append(sample)

        if len(samples) >= batch_size:
            training_session.add_all(samples)
            training_session.flush()
            total_saved += len(samples)
            samples = []

    if samples:
        training_session.add_all(samples)
        training_session.flush()
        total_saved += len(samples)

    training_session.commit()
    return total_saved