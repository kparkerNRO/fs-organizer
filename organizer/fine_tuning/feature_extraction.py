import json
from typing import Dict, List, Optional, Set

from sqlalchemy.orm import Session
from storage.index_models import Node
from storage.manager import NodeKind
from storage.training_models import LabelRun, TrainingSample
from utils.config import Config
from utils.text_processing import has_matching_token, tokenize_string
from fine_tuning.heuristic_classifier import COLLAB_MARKERS
from fine_tuning.utils import load_and_index_nodes, precompute_descendant_extensions


def _build_feature_text(
    node: Node,
    name_norm: str,
    parent: Optional[Node],
    grandparent: Optional[Node],
    sibling_names: List[str],
    child_names: List[str],
    extensions: List[str],
    flags: Dict[str, bool],
) -> str:
    """Construct the semi-structured text feature for the model."""
    flag_str = " ".join(f"{key}={int(val)}" for key, val in flags.items())
    return " | ".join(
        [
            f"gp:{grandparent.name if grandparent else ''}",
            f"p:{parent.name if parent else ''}",
            f"t:{name_norm}",
            f"depth:{node.depth}",
            "sibs:" + " ".join(sibling_names),
            "children:" + " ".join(child_names),
            "exts:" + " ".join(extensions),
            f"flags:{flag_str}",
        ]
    )


def _create_training_sample(
    node: Node,
    label_run: LabelRun,
    config: Config,
    nodes_by_id: Dict[int, Node],
    processed_name_by_id: Dict[int, str],
    children_by_parent: Dict[Optional[int], List[int]],
    descendant_exts: Dict[int, Set[str]],
    child_cap: int,
    sibling_cap: int,
    ext_cap: int,
) -> TrainingSample:
    """Create a single TrainingSample object for a given node."""
    name_norm = processed_name_by_id[node.node_id]
    parent_id = node.parent_node_id
    parent = nodes_by_id.get(parent_id) if parent_id is not None else None
    grandparent = nodes_by_id.get(parent.parent_node_id) if parent and parent.parent_node_id is not None else None

    child_ids = children_by_parent.get(node.node_id, [])
    sibling_ids = children_by_parent.get(parent_id, []) if parent_id else []
    
    child_names = sorted({processed_name_by_id[c] for c in child_ids})[:child_cap]
    sibling_names = sorted({processed_name_by_id[s] for s in sibling_ids if s != node.node_id})[:sibling_cap]

    child_token_bag = [t for cn in child_names for t in tokenize_string(cn)]
    sibling_token_bag = [t for sn in sibling_names for t in tokenize_string(sn)]

    flags = {
        "collab": has_matching_token(tokenize_string(node.name), COLLAB_MARKERS)
        or (parent is not None and has_matching_token(tokenize_string(parent.name), COLLAB_MARKERS)),
        "childMedia": has_matching_token(child_token_bag, config.media_types),
        "childVarHint": has_matching_token(child_token_bag, config.variant_types),
        "childFmt": any(cn.strip(".").lower() in config.format_types for cn in child_names),
        "sibVarHint": has_matching_token(sibling_token_bag, config.variant_types),
    }

    exts = sorted(descendant_exts[node.node_id])[:ext_cap]
    text = _build_feature_text(node, name_norm, parent, grandparent, sibling_names, child_names, exts, flags)

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
        has_collab_cue=flags["collab"],
        looks_like_format=config.is_media_type(node.name),
        child_has_media_type_cue=flags["childMedia"],
        child_has_variant_hint=flags["childVarHint"],
        child_has_format_cue=flags["childFmt"],
        sibling_has_variant_hint=flags["sibVarHint"],
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
    """
    Extract features for all nodes in a snapshot and save to training database.
    """
    nodes_by_id, processed_name_by_id, children_by_parent = load_and_index_nodes(index_session, snapshot_id)
    if not nodes_by_id:
        return 0

    descendant_exts = precompute_descendant_extensions(nodes_by_id, children_by_parent)

    samples: List[TrainingSample] = []
    total_saved = 0

    for node in nodes_by_id.values():
        is_zip_file = node.ext and node.ext.lower() in (".zip", "zip")
        if node.kind != NodeKind.DIR and not is_zip_file:
            continue

        sample = _create_training_sample(
            node, label_run, config, nodes_by_id, processed_name_by_id,
            children_by_parent, descendant_exts, child_cap, sibling_cap, ext_cap
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

    return total_saved