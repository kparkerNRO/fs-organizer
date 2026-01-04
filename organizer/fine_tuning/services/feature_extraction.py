import json
import logging
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from storage.index_models import Node
from storage.manager import StorageManager
from storage.training_models import LabelRun, TrainingSample
from utils.config import Config
from utils.text_processing import has_matching_token, tokenize_string

from fine_tuning.classifiers.heuristic_classifier import (
    COLLAB_MARKERS,
)
from fine_tuning.services.common import (
    FeatureNodeCore,
    extract_feature_nodes,
)

logger = logging.getLogger(__name__)


class FeatureExtractionConfigSettings(BaseModel):
    """Stable feature extraction configuration (loaded from config file)."""

    batch_size: int = Field(
        1000,
        description="Number of samples to insert per batch",
    )
    child_cap: int = Field(
        5,
        description="Maximum number of child folder names to include in features",
    )
    sibling_cap: int = Field(
        5,
        description="Maximum number of sibling folder names to include in features",
    )
    ext_cap: int = Field(
        10,
        description="Maximum number of file extensions to include in features",
    )


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


def extract_features_for_run(
    index_session: Session,
    training_session: Session,
    snapshot_id: int,
    app_config: Config,
    label_run: LabelRun,
    settings: FeatureExtractionConfigSettings,
) -> int:
    rows = list(
        index_session.execute(select(Node).where(Node.snapshot_id == snapshot_id)).scalars()
    )

    feature_nodes: list[FeatureNodeCore] = extract_feature_nodes(
        index_session=index_session,
        snapshot_id=snapshot_id,
        nodes=rows,
        max_siblings=settings.child_cap,
        max_descendents=settings.ext_cap,
        max_children=settings.child_cap,
    )

    samples: List[TrainingSample] = []
    total_saved = 0

    for feature_node in feature_nodes:
        node = feature_node.node
        parent = feature_node.parent
        grandparent = feature_node.grandparent
        name_norm = ""  # TODO
        # name_norm = processed_name_by_id[node.node_id]

        child_token_bag = [t for cn in feature_node.child_names for t in tokenize_string(cn)]
        sibling_token_bag = [t for sn in feature_node.sibling_names for t in tokenize_string(sn)]

        flags = {
            "collab": has_matching_token(tokenize_string(node.name), COLLAB_MARKERS)
            or (
                parent is not None
                and has_matching_token(tokenize_string(parent.name), COLLAB_MARKERS)
            ),
            "childMedia": has_matching_token(child_token_bag, app_config.media_types),
            "childVarHint": has_matching_token(child_token_bag, app_config.variant_types),
            "childFmt": any(
                cn.strip(".").lower() in app_config.format_types for cn in feature_node.child_names
            ),
            "sibVarHint": has_matching_token(sibling_token_bag, app_config.variant_types),
        }

        text = _build_feature_text(
            node,
            name_norm,
            parent,
            grandparent,
            feature_node.sibling_names,
            feature_node.child_names,
            feature_node.descendent_extentions,
            flags,
        )

        training_sample = TrainingSample(
            snapshot_id=feature_node.snapshot_id,
            node_id=feature_node.node.id,
            name_raw=feature_node.node.name,
            name_norm=name_norm,
            parent_name_norm=feature_node.parent_name,
            grandparent_name_norm=feature_node.grandparent_name,
            kind=feature_node.node.kind,
            file_source=feature_node.node.file_source,
            depth=int(feature_node.node.depth),
            child_names_topk_json=json.dumps(feature_node.child_names),
            sibling_names_topk_json=json.dumps(feature_node.sibling_names),
            descendant_file_exts_topk_json=json.dumps(feature_node.descendent_extentions),
            has_collab_cue=flags["collab"],
            looks_like_format=app_config.is_media_type(node.name),
            child_has_media_type_cue=flags["childMedia"],
            child_has_variant_hint=flags["childVarHint"],
            child_has_format_cue=flags["childFmt"],
            sibling_has_variant_hint=flags["sibVarHint"],
            text=text,
            label_run=label_run,
        )
        samples.append(training_sample)

        # Batch flush. TODO: does sqlalchemy do this better?
        if len(samples) >= settings.batch_size:
            training_session.add_all(samples)
            training_session.flush()
            total_saved += len(samples)
            samples = []
    if samples:
        training_session.add_all(samples)
        training_session.flush()
        total_saved += len(samples)

    return total_saved


def extract_features_from_snapshot(
    extraction_config: FeatureExtractionConfigSettings,
    manager: StorageManager,
    app_config: Config,
    snapshot_id: int,
) -> int:
    """
    Extract features for all nodes in a snapshot and save to training database.
    """
    with (
        manager.get_training_session() as training_session,
        manager.get_index_session(read_only=True) as index_session,
    ):
        label_run = LabelRun(snapshot_id=snapshot_id, label_source="unlabeled")
        training_session.add(label_run)
        training_session.flush()

        return extract_features_for_run(
            index_session,
            training_session,
            snapshot_id,
            app_config,
            label_run,
            extraction_config,
        )
