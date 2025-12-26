"""
Work through the stored file data, and extract features for model training

Based on the discussion here: https://chatgpt.com/share/694e0924-de18-8007-bedc-da05b1cac83c

"""

from __future__ import annotations
from utils.filename_utils import clean_filename, has_close_match

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from utils.config import Config
from sqlalchemy import select
from sqlalchemy.orm import Session

from storage.index_models import Node
from storage.manager import NodeKind, FileSource


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

def normalize_string(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def token_ready_strings(s: str) -> List[str]:
    return TOKEN_RE.findall(normalize_string(s))


def _processed_name(name: str) -> str:
    if name.lower().endswith(".zip"):
        return clean_filename(name[:-4])
    return clean_filename(name)


def has_any_token(token_list: List[str], cue_set: Set[str]) -> bool:
    return any(has_close_match(t, token_list) in cue_set for t in token_list)


@dataclass
class NodeFeatures:
    # identity
    node_id: int
    snapshot_id: int
    kind: str
    file_source: str

    # raw context
    name_raw: str
    name_norm: str
    parent_name_norm: Optional[str]
    grandparent_name_norm: Optional[str]
    depth: int

    # structural context
    child_names_topk: List[str]
    sibling_names_topk: List[str]
    descendant_file_exts_topk: List[
        str
    ]  # derived summary (open vocab but usually small)

    # cue flags
    has_collab_cue: bool
    looks_like_format: bool
    child_has_media_type_cue: bool
    child_has_variant_hint: bool
    child_has_format_cue: bool
    sibling_has_variant_hint: bool

    # serialized model text input (optional but convenient)
    text: str

def extract_features(
    index_session: Session,
    work_session: Session,
    snapshot_id: int,
    config: Config,
    *,
    child_cap: int = 20,
    sibling_cap: int = 20,
    ext_cap: int = 12,
) -> List[NodeFeatures] | None:
    """
    Builds feature rows for ALL nodes in a snapshot.

    Notes for your ZIP modeling:
    - Treat nodes with file_source='zip_file' as containers (even if kind='file').
    - ZIP content nodes already have parent_node_id pointing at that 'zip_file' node,
      so adjacency building works naturally.
    """

    # 1) Load nodes for snapshot
    rows = index_session.execute(
        select(Node)
        .where(Node.snapshot_id == snapshot_id)
    ).scalars()

    # 2) Index nodes + build parent->children adjacency
    nodes_by_id: Dict[int, Node] = {}
    processed_name_by_id : dict[int, str] = {}
    children_by_parent: Dict[Optional[int], List[int]] = {}

    for r in rows:
        # node_id, snap_id, parent_id, kind, name, ext, depth, file_source = r
        nodes_by_id[r.node_id] = r
        processed_name_by_id[r.node_id] = _processed_name(r.name)
        children_by_parent.setdefault(r.parent_node_id, []).append(r.node_id)

    # Helper: container-ness (dirs + zip_file nodes act like dirs)
    def is_container(n: Node) -> bool:
        return n.kind == NodeKind.DIR or n.file_source == FileSource.ZIP_FILE

    # 3) Precompute descendant file extensions per node (DP over depths)
    #    We'll accumulate a small set per node, then cap when serializing.
    #    This is optional but often helpful; cheap enough to compute here.
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

    # 4) Build features per node (both files and folders)
    features: List[NodeFeatures] = []

    for nid, n in nodes_by_id.items():
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
        child_token_bag = [t for cn in child_names for t in token_ready_strings(cn)]
        sibling_token_bag = [t for sn in sibling_names for t in token_ready_strings(sn)]

        child_has_media_type_cue = has_any_token(child_token_bag, config.media_types)
        child_has_variant_hint = has_any_token(child_token_bag, config.variant_types)
        child_has_format_cue = any(cn.strip(".").lower() in config.format_types for cn in child_names)

        sibling_has_variant_hint = has_any_token(sibling_token_bag, config.variant_types)

        # collab cue from self or parent
        has_collab_cue = has_any_token(token_ready_strings(n.name), config.collab_markers) or (
            parent is not None and has_any_token(token_ready_strings(parent.name), config.collab_markers)
        )

        looks_like_format = config.is_media_type(n.name)

        # descendant exts summary
        exts = sorted(descendant_exts[nid])[:ext_cap]

        text = " | ".join([
            f"gp:{grandparent.name if grandparent else ''}",
            f"p:{parent.name if parent else ''}",
            f"t:{name_norm}",
            f"depth:{n.depth}",
            "sibs:" + " ".join(sibling_names),
            "children:" + " ".join(child_names),
            "exts:" + " ".join(exts),
            f"flags:collab={int(has_collab_cue)} childMedia={int(child_has_media_type_cue)} childVarHint={int(child_has_variant_hint)} childFmt={int(child_has_format_cue)}",
        ])

        features.append(
            NodeFeatures(
                node_id=n.node_id,
                snapshot_id=n.snapshot_id,
                kind=n.kind,
                file_source=n.file_source,
                name_raw=n.name,
                name_norm=name_norm,
                parent_name_norm=parent.name if parent else None,
                grandparent_name_norm=grandparent.name if grandparent else None,
                depth=int(n.depth),
                child_names_topk=child_names,
                sibling_names_topk=sibling_names,
                descendant_file_exts_topk=exts,
                has_collab_cue=has_collab_cue,
                looks_like_format=looks_like_format,
                child_has_media_type_cue=child_has_media_type_cue,
                child_has_variant_hint=child_has_variant_hint,
                child_has_format_cue=child_has_format_cue,
                sibling_has_variant_hint=sibling_has_variant_hint,
                text=text
            )  
        )

    return features
