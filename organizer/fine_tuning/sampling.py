"""Helper functions for training data generation and management.

This module provides utilities for:
- Selecting diverse training samples from snapshots
- Writing and reading CSV files for manual labeling
- Validating label data before storing in training database
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

from sqlalchemy import select
from sqlalchemy.orm import Session
from storage.index_models import Node
from storage.manager import NodeKind
from utils.text_processing import char_trigrams, jaccard_similarity

from fine_tuning.taxonomy import LABELS_LEGACY

# Valid labels for leaf folder classification (legacy taxonomy)
VALID_LABELS = LABELS_LEGACY

# CSV column schema for training samples
CSV_COLUMNS = [
    "snapshot_id",
    "node_id",
    "name",
    "rel_path",
    "depth",
    "parent_name",
    "grandparent_name",
    "num_children",
    "child_names_sample",
    "sibling_names_sample",
    "file_extensions",
    "heuristic_label",  # Heuristic prediction
    "heuristic_confidence",  # Confidence score
    "heuristic_reason",  # Reasoning
    "label",  # Manual label (to be filled)
]


def select_training_samples(
    session: Session,
    snapshot_id: int,
    sample_size: int,
    min_depth: int,
    max_depth: int,
    diversity_factor: float,
) -> List[Node]:
    """Select diverse training samples from a snapshot using stratified sampling.

    Strategy:
    1. Query folders only (kind='dir') within depth range
    2. Stratify by depth level
    3. Cluster similar names using character trigrams
    4. Sample diverse folders from each depth stratum

    Args:
        session: SQLAlchemy session for index.db
        snapshot_id: Snapshot to sample from
        sample_size: Target number of samples
        min_depth: Minimum depth to sample from
        max_depth: Maximum depth to sample from
        diversity_factor: Balance between random and diverse (0-1, higher=more diverse)

    Returns:
        List of selected Node objects
    """
    # Query folders in depth range
    nodes = (
        session.execute(
            select(Node)
            .where(Node.snapshot_id == snapshot_id)
            .where(Node.kind == NodeKind.DIR.value)
            .where(Node.depth >= min_depth)
            .where(Node.depth <= max_depth)
        )
        .scalars()
        .all()
    )

    if not nodes:
        return []

    # Group by depth
    by_depth: Dict[int, List[Node]] = defaultdict(list)
    for node in nodes:
        by_depth[node.depth].append(node)

    # Calculate samples per depth proportionally
    total_nodes = len(nodes)
    samples_per_depth: Dict[int, int] = {}
    min_per_depth = 10  # Minimum samples per depth if available

    for depth, depth_nodes in by_depth.items():
        proportion = len(depth_nodes) / total_nodes
        allocated = max(min_per_depth, int(sample_size * proportion))
        # Cap at available nodes
        samples_per_depth[depth] = min(allocated, len(depth_nodes))

    # Adjust if total exceeds sample_size
    total_allocated = sum(samples_per_depth.values())
    if total_allocated > sample_size:
        # Scale down proportionally
        scale_factor = sample_size / total_allocated
        for depth in samples_per_depth:
            samples_per_depth[depth] = max(1, int(samples_per_depth[depth] * scale_factor))

    # Sample from each depth with diversity clustering
    selected: List[Node] = []

    for depth, num_samples in samples_per_depth.items():
        depth_nodes = by_depth[depth]
        if len(depth_nodes) <= num_samples:
            selected.extend(depth_nodes)
            continue

        # Cluster by name similarity
        clusters = _cluster_by_similarity(depth_nodes, threshold=0.4)

        # Sample from clusters to maximize diversity
        sampled = _sample_from_clusters(clusters, num_samples, diversity_factor)
        selected.extend(sampled)

    return selected[:sample_size]


def _cluster_by_similarity(nodes: List[Node], threshold: float) -> List[List[Node]]:
    """Cluster nodes by name similarity using character trigrams.

    Args:
        nodes: List of nodes to cluster
        threshold: Jaccard similarity threshold (0-1)

    Returns:
        List of clusters (each cluster is a list of nodes)
    """
    if not nodes:
        return []

    # Precompute trigrams
    trigrams = [char_trigrams(node.name.lower()) for node in nodes]

    clusters: List[List[int]] = []  # indices into nodes list
    assigned = set()

    for i, node in enumerate(nodes):
        if i in assigned:
            continue

        # Start new cluster
        cluster = [i]
        assigned.add(i)

        # Find similar nodes
        for j in range(i + 1, len(nodes)):
            if j in assigned:
                continue

            sim = jaccard_similarity(trigrams[i], trigrams[j])
            if sim >= threshold:
                cluster.append(j)
                assigned.add(j)

        clusters.append(cluster)

    # Convert indices to nodes
    return [[nodes[idx] for idx in cluster] for cluster in clusters]


def _sample_from_clusters(
    clusters: List[List[Node]], num_samples: int, diversity_factor: float
) -> List[Node]:
    """Sample nodes from clusters to maximize diversity.

    Args:
        clusters: List of node clusters
        num_samples: Number of samples to select
        diversity_factor: Balance between random and diverse (0-1)

    Returns:
        List of selected nodes
    """
    if not clusters:
        return []

    # Sort clusters by size (descending)
    sorted_clusters: List[Node] = sorted(clusters, key=len, reverse=True)

    # Calculate max samples per cluster
    max_per_cluster = max(1, int(3 * (1 - diversity_factor) + 1))

    selected: List[Node] = []
    cluster_idx = 0

    while len(selected) < num_samples and cluster_idx < len(sorted_clusters):
        cluster = sorted_clusters[cluster_idx]

        # Take up to max_per_cluster from this cluster
        take = min(max_per_cluster, len(cluster), num_samples - len(selected))
        selected.extend(cluster[:take])

        # Remove sampled nodes from cluster
        sorted_clusters[cluster_idx] = cluster[take:]

        # Move to next cluster (round-robin)
        cluster_idx = (cluster_idx + 1) % len(sorted_clusters)

        # Remove empty clusters
        sorted_clusters = [c for c in sorted_clusters if c]
        if not sorted_clusters:
            break

        cluster_idx = cluster_idx % len(sorted_clusters)

    return selected


def write_sample_csv(
    output_path: Path,
    nodes: List[Node],
    session: Session,
    snapshot_id: int,
    child_sample_size: int = 5,
    sibling_sample_size: int = 5,
    ext_sample_size: int = 10,
    use_heuristic: bool = True,
    heuristic_taxonomy: str = "v2",
) -> None:
    """Write training sample CSV with context columns.

    Args:
        output_path: Path to output CSV file
        nodes: List of nodes to write
        session: SQLAlchemy session for index.db
        snapshot_id: Snapshot ID
        child_sample_size: Max child names to sample
        sibling_sample_size: Max sibling names to sample
        ext_sample_size: Max file extensions to sample
        use_heuristic: Whether to include heuristic classifier predictions
        heuristic_taxonomy: Taxonomy for heuristic classifier ('v1' or 'v2')
    """
    # Load all nodes for snapshot to build adjacency
    all_nodes = session.execute(select(Node).where(Node.snapshot_id == snapshot_id)).scalars().all()

    # Build node lookup and parent->children map
    nodes_by_id: Dict[int, Node] = {n.node_id: n for n in all_nodes}
    children_by_parent: Dict[int, List[Node]] = defaultdict(list)

    for node in all_nodes:
        if node.parent_node_id is not None:
            children_by_parent[node.parent_node_id].append(node)

    # Collect file extensions recursively
    def collect_extensions(node_id: int) -> Set[str]:
        """Recursively collect file extensions from descendants."""
        exts = set()
        for child in children_by_parent.get(node_id, []):
            if child.kind == NodeKind.FILE.value and child.ext:
                exts.add(child.ext.lstrip(".").lower())
            exts.update(collect_extensions(child.node_id))
        return exts

    # Initialize heuristic classifier if requested
    heuristic_classifier = None
    if use_heuristic:
        try:
            from utils.config import Config

            from fine_tuning.heuristic_classifier import HeuristicClassifier

            config = Config()
            heuristic_classifier = HeuristicClassifier(config, taxonomy=heuristic_taxonomy)
        except Exception as e:
            print(f"Warning: Could not initialize heuristic classifier: {e}")
            use_heuristic = False

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for node in nodes:
            # Get parent and grandparent
            parent = nodes_by_id.get(node.parent_node_id) if node.parent_node_id else None
            grandparent = (
                nodes_by_id.get(parent.parent_node_id) if parent and parent.parent_node_id else None
            )

            # Get children (folders only)
            children = [
                c for c in children_by_parent.get(node.node_id, []) if c.kind == NodeKind.DIR.value
            ]
            child_names = sorted({c.name for c in children})[:child_sample_size]

            # Get siblings (folders only)
            if parent:
                siblings = [
                    c
                    for c in children_by_parent.get(parent.node_id, [])
                    if c.kind == NodeKind.DIR.value and c.node_id != node.node_id
                ]
                sibling_names = sorted({s.name for s in siblings})[:sibling_sample_size]
            else:
                sibling_names = []

            # Get file extensions
            extensions = sorted(collect_extensions(node.node_id))[:ext_sample_size]

            # Run heuristic classifier if enabled
            heuristic_label = ""
            heuristic_confidence = ""
            heuristic_reason = ""

            if heuristic_classifier:
                result = heuristic_classifier.classify(
                    name=node.name,
                    depth=node.depth,
                    parent_name=parent.name if parent else None,
                    children_names=child_names,
                    sibling_names=sibling_names,
                    file_extensions=extensions,
                )
                heuristic_label = result.label
                heuristic_confidence = f"{result.confidence:.3f}"
                heuristic_reason = result.reason

            # Write row
            writer.writerow(
                {
                    "snapshot_id": snapshot_id,
                    "node_id": node.node_id,
                    "name": node.name,
                    "rel_path": node.rel_path,
                    "depth": node.depth,
                    "parent_name": parent.name if parent else "",
                    "grandparent_name": grandparent.name if grandparent else "",
                    "num_children": len(children),
                    "child_names_sample": json.dumps(child_names),
                    "sibling_names_sample": json.dumps(sibling_names),
                    "file_extensions": json.dumps(extensions),
                    "heuristic_label": heuristic_label,
                    "heuristic_confidence": heuristic_confidence,
                    "heuristic_reason": heuristic_reason,
                    "label": "",  # Empty for manual labeling
                }
            )


def read_classification_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Read and parse classification CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of row dictionaries with parsed fields

    Raises:
        ValueError: If CSV format is invalid
    """
    rows: List[Dict[str, Any]] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Validate columns
        if not reader.fieldnames:
            raise ValueError("CSV file is empty or has no header")

        missing = set(CSV_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        for i, row in enumerate(reader, start=2):  # Line 2 is first data row
            try:
                # Parse numeric fields
                snapshot_id = int(row["snapshot_id"])
                node_id = int(row["node_id"])
                depth = int(row["depth"])
                num_children = int(row["num_children"])

                # Parse JSON fields
                child_names = json.loads(row["child_names_sample"])
                sibling_names = json.loads(row["sibling_names_sample"])
                extensions = json.loads(row["file_extensions"])

                rows.append(
                    {
                        "snapshot_id": snapshot_id,
                        "node_id": node_id,
                        "name": row["name"],
                        "rel_path": row["rel_path"],
                        "depth": depth,
                        "parent_name": row["parent_name"],
                        "grandparent_name": row["grandparent_name"],
                        "num_children": num_children,
                        "child_names_sample": child_names,
                        "sibling_names_sample": sibling_names,
                        "file_extensions": extensions,
                        "label": row["label"].strip(),
                    }
                )
            except (ValueError, json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Error parsing CSV row {i}: {e}")

    return rows


def validate_all_labels_present(rows: List[Dict[str, Any]]) -> None:
    """Validate that all rows have non-empty labels.

    Args:
        rows: List of parsed CSV rows

    Raises:
        ValueError: If any rows are missing labels
    """
    unlabeled = [i + 2 for i, row in enumerate(rows) if not row["label"]]

    if unlabeled:
        if len(unlabeled) <= 10:
            rows_str = ", ".join(map(str, unlabeled))
        else:
            rows_str = ", ".join(map(str, unlabeled[:10])) + f", ... ({len(unlabeled)} total)"

        raise ValueError(
            f"Found {len(unlabeled)} rows with missing labels.\n"
            f"Rows: {rows_str}\n"
            f"Please fill in the 'label' column for all rows."
        )


def validate_label_values(rows: List[Dict[str, Any]]) -> None:
    """Validate that all labels are valid.

    Args:
        rows: List of parsed CSV rows

    Raises:
        ValueError: If any labels are invalid
    """
    invalid_labels = defaultdict(list)

    for i, row in enumerate(rows, start=2):
        label = row["label"]
        if label and label not in VALID_LABELS:
            invalid_labels[label].append(i)

    if invalid_labels:
        error_msg = [f"Found invalid labels in CSV:"]
        for label, row_nums in sorted(invalid_labels.items()):
            if len(row_nums) <= 5:
                rows_str = ", ".join(map(str, row_nums))
            else:
                rows_str = ", ".join(map(str, row_nums[:5])) + f", ... ({len(row_nums)} total)"
            error_msg.append(f"  '{label}': rows {rows_str}")

        error_msg.append(f"\nValid labels are: {', '.join(sorted(VALID_LABELS))}")
        raise ValueError("\n".join(error_msg))
