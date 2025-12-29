import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

from sqlalchemy.orm import Session, select
from storage.index_models import Node
from storage.manager import NodeKind
from utils.config import get_config
from utils.text_processing import char_trigrams, jaccard_similarity

from fine_tuning.heuristic_classifier import HeuristicClassifier
from fine_tuning.utils import load_and_index_nodes, precompute_descendant_extensions


# CSV column schema for training samples
CSV_COLUMNS = [
    "snapshot_id",
    "node_id",
    "rel_path",
    "depth",
    "parent_name",
    "grandparent_name",
    "num_children",
    "child_names_sample",
    "sibling_names_sample",
    "file_extensions",
    "name",
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
    """Select diverse training samples from a snapshot using stratified sampling."""
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

    by_depth: Dict[int, List[Node]] = defaultdict(list)
    for node in nodes:
        by_depth[node.depth].append(node)

    total_nodes = len(nodes)
    samples_per_depth: Dict[int, int] = {}
    min_per_depth = 10
    for depth, depth_nodes in by_depth.items():
        proportion = len(depth_nodes) / total_nodes
        allocated = max(min_per_depth, int(sample_size * proportion))
        samples_per_depth[depth] = min(allocated, len(depth_nodes))

    total_allocated = sum(samples_per_depth.values())
    if total_allocated > sample_size:
        scale_factor = sample_size / total_allocated
        for depth in samples_per_depth:
            samples_per_depth[depth] = max(
                1, int(samples_per_depth[depth] * scale_factor)
            )

    selected: List[Node] = []
    for depth, num_samples in samples_per_depth.items():
        depth_nodes = by_depth[depth]
        if len(depth_nodes) <= num_samples:
            selected.extend(depth_nodes)
            continue
        clusters = _cluster_by_similarity(depth_nodes, threshold=0.4)
        sampled = _sample_from_clusters(clusters, num_samples, diversity_factor)
        selected.extend(sampled)

    return selected[:sample_size]


def _cluster_by_similarity(nodes: List[Node], threshold: float) -> List[List[Node]]:
    """Cluster nodes by name similarity using character trigrams."""
    if not nodes:
        return []
    trigrams = [char_trigrams(node.name.lower()) for node in nodes]
    clusters: List[List[int]] = []
    assigned = set()
    for i in range(len(nodes)):
        if i in assigned:
            continue
        cluster = [i]
        assigned.add(i)
        for j in range(i + 1, len(nodes)):
            if j in assigned:
                continue
            sim = jaccard_similarity(trigrams[i], trigrams[j])
            if sim >= threshold:
                cluster.append(j)
                assigned.add(j)
        clusters.append(cluster)
    return [[nodes[idx] for idx in cluster] for cluster in clusters]


def _sample_from_clusters(
    clusters: List[List[Node]], num_samples: int, diversity_factor: float
) -> List[Node]:
    """Sample nodes from clusters to maximize diversity."""
    if not clusters:
        return []
    
    # Flatten clusters into a list of nodes, ensuring at least one from each
    selected: List[Node] = []
    # Take one from each cluster to guarantee diversity
    for cluster in clusters:
        if cluster:
            selected.append(cluster.pop(0))

    # Fill remaining from all available nodes
    remaining_nodes = [node for cluster in clusters for node in cluster]
    
    import random
    random.shuffle(remaining_nodes)

    needed = num_samples - len(selected)
    if needed > 0:
        selected.extend(remaining_nodes[:needed])

    return selected[:num_samples]


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
    """Write training sample CSV with context columns, efficiently."""
    nodes_by_id, _, children_by_parent = load_and_index_nodes(session, snapshot_id)
    descendant_exts = precompute_descendant_extensions(nodes_by_id, children_by_parent)

    heuristic_classifier = None
    if use_heuristic:
        try:
            config = get_config()
            heuristic_classifier = HeuristicClassifier(
                config, taxonomy=heuristic_taxonomy
            )
        except Exception as e:
            print(f"Warning: Could not initialize heuristic classifier: {e}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for node in nodes:
            parent = nodes_by_id.get(node.parent_node_id) if node.parent_node_id else None
            grandparent = nodes_by_id.get(parent.parent_node_id) if parent and parent.parent_node_id else None

            child_nodes = [nodes_by_id[cid] for cid in children_by_parent.get(node.node_id, []) if cid in nodes_by_id and nodes_by_id[cid].kind == NodeKind.DIR]
            child_names = sorted({c.name for c in child_nodes})[:child_sample_size]

            if parent:
                sibling_nodes = [nodes_by_id[sid] for sid in children_by_parent.get(parent.node_id, []) if sid != node.node_id and sid in nodes_by_id and nodes_by_id[sid].kind == NodeKind.DIR]
                sibling_names = sorted({s.name for s in sibling_nodes})[:sibling_sample_size]
            else:
                sibling_names = []

            extensions = sorted(descendant_exts.get(node.node_id, set()))[:ext_sample_size]

            row = {
                "snapshot_id": snapshot_id,
                "node_id": node.node_id,
                "rel_path": node.rel_path,
                "depth": node.depth,
                "parent_name": parent.name if parent else "",
                "grandparent_name": grandparent.name if grandparent else "",
                "num_children": len(child_nodes),
                "child_names_sample": json.dumps(child_names),
                "sibling_names_sample": json.dumps(sibling_names),
                "file_extensions": json.dumps(extensions),
                "name": node.name,
                "heuristic_label": "",
                "heuristic_confidence": "",
                "heuristic_reason": "",
                "label": "",
            }

            if heuristic_classifier:
                result = heuristic_classifier.classify(
                    name=node.name,
                    depth=node.depth,
                    parent_name=parent.name if parent else None,
                    children_names=child_names,
                    sibling_names=sibling_names,
                    file_extensions=extensions,
                )
                row["heuristic_label"] = result.label
                row["heuristic_confidence"] = f"{result.confidence:.3f}"
                row["heuristic_reason"] = result.reason

            writer.writerow(row)


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

        # Check only for the presence of 'label', 'snapshot_id', and 'node_id'
        required_cols = {"label", "snapshot_id", "node_id"}
        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        for i, row in enumerate(reader, start=2):  # Line 2 is first data row
            try:
                # Parse required fields
                snapshot_id = int(row["snapshot_id"])
                node_id = int(row["node_id"])
                label = row["label"].strip()
                
                # Create a flexible dictionary from the row
                parsed_row = {
                    "snapshot_id": snapshot_id,
                    "node_id": node_id,
                    "label": label,
                }
                # Add all other columns from the CSV dynamically
                for col, value in row.items():
                    if col not in parsed_row:
                        parsed_row[col] = value

                rows.append(parsed_row)

            except (ValueError, KeyError) as e:
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
            rows_str = (
                ", ".join(map(str, unlabeled[:10])) + f", ... ({len(unlabeled)} total)"
            )

        raise ValueError(
            f"Found {len(unlabeled)} rows with missing labels.\n"
            f"Rows: {rows_str}\n"
            f"Please fill in the 'label' column for all rows."
        )


def validate_label_values(rows: List[Dict[str, Any]], valid_labels) -> None:
    """Validate that all labels are valid.

    Args:
        rows: List of parsed CSV rows

    Raises:
        ValueError: If any labels are invalid
    """
    invalid_labels = defaultdict(list)

    for i, row in enumerate(rows, start=2):
        label = row["label"]
        if label and label not in valid_labels:
            invalid_labels[label].append(i)

    if invalid_labels:
        error_msg = ["Found invalid labels in CSV:"]
        for label, row_nums in sorted(invalid_labels.items()):
            if len(row_nums) <= 5:
                rows_str = ", ".join(map(str, row_nums))
            else:
                rows_str = (
                    ", ".join(map(str, row_nums[:5])) + f", ... ({len(row_nums)} total)"
                )
            error_msg.append(f"  '{label}': rows {rows_str}")

        error_msg.append(f"\nValid labels are: {', '.join(sorted(valid_labels))}")
        raise ValueError("\n".join(error_msg))