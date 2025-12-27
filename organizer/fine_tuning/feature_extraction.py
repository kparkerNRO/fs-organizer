"""
Work through the stored file data, and extract features for model training

Based on the discussion here: https://chatgpt.com/share/694e0924-de18-8007-bedc-da05b1cac83c

"""

import json
from typing import Dict, List, Optional, Set

from sqlalchemy import select
from sqlalchemy.orm import Session
from storage.index_models import Node
from storage.manager import NodeKind
from storage.training_models import TrainingSample
from utils.config import Config
from utils.filename_processing import _processed_name
from utils.text_processing import has_matching_token, normalize_string, tokenize_string
from fine_tuning.heuristic_classifier import COLLAB_MARKERS

def extract_features(
    index_session: Session,
    training_session: Session,
    snapshot_id: int,
    config: Config,
    *,
    child_cap: int = 20,
    sibling_cap: int = 20,
    ext_cap: int = 12,
    batch_size: int = 1000,
) -> int:
    """
    Extract features for all nodes in a snapshot and save to training database.

    Args:
        index_session: SQLAlchemy session for index.db
        training_session: SQLAlchemy session for training.db
        snapshot_id: The snapshot to extract features from
        config: Configuration with cue markers and types
        child_cap: Maximum number of child names to store
        sibling_cap: Maximum number of sibling names to store
        ext_cap: Maximum number of file extensions to store
        batch_size: Number of samples to insert per batch

    Returns:
        Number of training samples created

    Notes:
        - Treat nodes with file_source='zip_file' as containers (even if kind='file')
        - ZIP content nodes already have parent_node_id pointing at that 'zip_file' node
    """

    # Load nodes 
    rows = index_session.execute(select(Node).where(Node.snapshot_id == snapshot_id)).scalars()

    # TODO: compute frequencies of words

    # Index nodes + build parent->children adjacency
    nodes_by_id: Dict[int, Node] = {}
    processed_name_by_id: dict[int, str] = {}
    children_by_parent: Dict[Optional[int], List[int]] = {}

    for r in rows:
        nodes_by_id[r.node_id] = r
        processed_name_by_id[r.node_id] = _processed_name(r.name)
        children_by_parent.setdefault(r.parent_node_id, []).append(r.node_id)

    # Precompute descendant file extensions per node (DP over depths)
    depth_groups: Dict[int, List[int]] = {}
    max_depth = 0
    for node_id, n in nodes_by_id.items():
        d = int(n.depth)
        depth_groups.setdefault(d, []).append(node_id)
        if d > max_depth:
            max_depth = d

    descendant_exts: Dict[int, Set[str]] = {nid: set() for nid in nodes_by_id.keys()}

    # bottom-up: deepest first
    for d in range(max_depth, -1, -1):
        for nid in depth_groups.get(d, []):
            n = nodes_by_id[nid]
            # If this is a file, add its ext
            if n.kind == NodeKind.FILE and n.ext:
                descendant_exts[nid].add(normalize_string(n.ext).strip("."))
            # Add children exts
            for cid in children_by_parent.get(nid, []):
                descendant_exts[nid].update(descendant_exts[cid])

    # 4) Build features and save to database in batches
    samples: List[TrainingSample] = []
    total_saved = 0

    for nid, n in nodes_by_id.items():
        # Only process dirs for now
        if n.kind != NodeKind.DIR:
            continue

        name_norm = processed_name_by_id[nid]
        parent_id = n.parent_node_id
        parent = nodes_by_id.get(parent_id) if parent_id is not None else None
        grandparent = (
            nodes_by_id.get(parent.parent_node_id)
            if parent and parent.parent_node_id is not None
            else None
        )

        # children + siblings (names, normalized)
        child_ids = children_by_parent.get(nid, [])
        sibling_ids = (
            [sid for sid in children_by_parent.get(parent_id, []) if sid != nid]
            if parent_id in children_by_parent
            else []
        )

        child_names = sorted({processed_name_by_id[c] for c in child_ids})[:child_cap]
        sibling_names = sorted({processed_name_by_id[s] for s in sibling_ids})[:sibling_cap]

        # cues from children/siblings
        child_token_bag = [t for cn in child_names for t in tokenize_string(cn)]
        sibling_token_bag = [t for sn in sibling_names for t in tokenize_string(sn)]

        child_has_media_type_cue = has_matching_token(child_token_bag, config.media_types)
        child_has_variant_hint = has_matching_token(child_token_bag, config.variant_types)
        child_has_format_cue = any(
            cn.strip(".").lower() in config.format_types for cn in child_names
        )

        sibling_has_variant_hint = has_matching_token(sibling_token_bag, config.variant_types)

        # collab cue from self or parent
        has_collab_cue = has_matching_token(tokenize_string(n.name), COLLAB_MARKERS) or (
            parent is not None and has_matching_token(tokenize_string(parent.name), COLLAB_MARKERS)
        )

        looks_like_format = config.is_media_type(n.name)

        # descendant exts summary
        exts = sorted(descendant_exts[nid])[:ext_cap]

        text = " | ".join(
            [
                f"gp:{grandparent.name if grandparent else ''}",
                f"p:{parent.name if parent else ''}",
                f"t:{name_norm}",
                f"depth:{n.depth}",
                "sibs:" + " ".join(sibling_names),
                "children:" + " ".join(child_names),
                "exts:" + " ".join(exts),
                f"flags:collab={int(has_collab_cue)} childMedia={int(child_has_media_type_cue)} childVarHint={int(child_has_variant_hint)} childFmt={int(child_has_format_cue)}",
            ]
        )

        sample = TrainingSample(
            snapshot_id=n.snapshot_id,
            node_id=n.node_id,
            name_raw=n.name,
            name_norm=name_norm,
            parent_name_norm=parent.name if parent else None,
            grandparent_name_norm=grandparent.name if grandparent else None,
            kind=n.kind,
            file_source=n.file_source,
            depth=int(n.depth),
            child_names_topk_json=json.dumps(child_names),
            sibling_names_topk_json=json.dumps(sibling_names),
            descendant_file_exts_topk_json=json.dumps(exts),
            has_collab_cue=has_collab_cue,
            looks_like_format=looks_like_format,
            child_has_media_type_cue=child_has_media_type_cue,
            child_has_variant_hint=child_has_variant_hint,
            child_has_format_cue=child_has_format_cue,
            sibling_has_variant_hint=sibling_has_variant_hint,
            text=text,
        )

        samples.append(sample)

        # Batch insert
        if len(samples) >= batch_size:
            training_session.add_all(samples)
            training_session.flush()
            total_saved += len(samples)
            samples = []

    # Insert remaining samples
    if samples:
        training_session.add_all(samples)
        training_session.flush()
        total_saved += len(samples)

    training_session.commit()
    return total_saved
