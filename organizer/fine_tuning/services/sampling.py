import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fine_tuning.classifiers.heuristic_classifier import HeuristicClassifier
from fine_tuning.services.common import (
    extract_feature_nodes,
)
from fine_tuning.services.feature_extraction import (
    extract_features_for_run,
)
from fine_tuning.taxonomy import get_labels
from pydantic import Field
from pydantic.v1 import BaseSettings
from sqlalchemy import select
from sqlalchemy.orm import Session
from storage.index_models import Node
from storage.manager import NodeKind, StorageManager
from storage.training_models import LabelRun, TrainingSample
from utils.config import get_config
from utils.text_processing import char_trigrams, jaccard_similarity

logger = logging.getLogger(__name__)

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


class GenerateSamplesSettings(BaseSettings):
    """Settings for the 'generate-samples' command."""

    sample_size: int = Field(
        800,
        description="Number of samples to generate",
    )
    min_depth: int = Field(
        1,
        description="Minimum folder depth to sample from",
    )
    max_depth: int = Field(
        10,
        description="Maximum folder depth to sample from",
    )
    diversity_factor: float = Field(
        0.7,
        description="Balance between random and diverse sampling (0-1, higher=more diverse)",
    )
    use_heuristic: bool = Field(
        True,
        description="Include heuristic classifier predictions in CSV output",
    )
    heuristic_taxonomy: str = Field(
        "v2",
        description="Taxonomy for heuristic classifier (v1 or v2)",
    )


class ApplyClassificationsSettings(BaseSettings):
    """Settings for the 'apply-classifications' command."""

    labeler: str = Field(
        "manual",
        description="Name of the labeler (e.g., 'manual', 'human-v1')",
    )
    split: Optional[str] = Field(
        None,
        description="Data split: 'train', 'validation', or 'test'",
    )
    validate_only: bool = Field(
        False,
        description="Only validate CSV without writing to database",
    )


###############################
# Select samples
###############################


def select_training_samples(
    session: Session,
    snapshot_id: int,
    sample_size: int,
    min_depth: int,
    max_depth: int,
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
        sampled = _sample_from_clusters(clusters, num_samples)
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


def _sample_from_clusters(clusters: List[List[Node]], num_samples: int) -> List[Node]:
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

    heuristic_classifier = None
    if use_heuristic:
        try:
            config = get_config()
            heuristic_classifier = HeuristicClassifier(
                config, taxonomy=heuristic_taxonomy
            )
        except Exception as e:
            logger.warning(f"Could not initialize heuristic classifier: {e}")

    feature_nodes = extract_feature_nodes(
        index_session=session,
        snapshot_id=snapshot_id,
        nodes=nodes,
        max_siblings=sibling_sample_size,
        max_descendents=ext_sample_size,
        max_children=child_sample_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for feature_node in feature_nodes:
            row = {
                "snapshot_id": snapshot_id,
                "node_id": feature_node.node.node_id,
                "rel_path": feature_node.node.rel_path,
                "depth": feature_node.node.depth,
                "parent_name": feature_node.parent_name,
                "grandparent_name": feature_node.grandparent_name,
                "num_children": len(feature_node.child_nodes),
                "child_names_sample": json.dumps(feature_node.child_names),
                "sibling_names_sample": json.dumps(feature_node.sibling_names),
                "file_extensions": json.dumps(feature_node.descendent_extentions),
                "name": feature_node.node.name,
                "heuristic_label": "",
                "heuristic_confidence": "",
                "heuristic_reason": "",
                "label": "",
            }

            if heuristic_classifier:
                result = heuristic_classifier.classify(
                    name=feature_node.node.name,
                    depth=feature_node.node.depth,
                    parent_name=feature_node.parent.name,
                    children_names=feature_node.child_names,
                    sibling_names=feature_node.sibling_names,
                    file_extensions=feature_node.descendent_extentions,
                )
                row["heuristic_label"] = result.label
                row["heuristic_confidence"] = f"{result.confidence:.3f}"
                row["heuristic_reason"] = result.reason

            writer.writerow(row)


def generate_sample_data(
    settings: GenerateSamplesSettings,
    manager: StorageManager,
    snapshot_id: int,
    output_csv: Path,
) -> int:
    with manager.get_index_session(read_only=True) as session:
        samples = select_training_samples(
            session=session,
            snapshot_id=snapshot_id,
            sample_size=settings.sample_size,
            min_depth=settings.min_depth,
            max_depth=settings.max_depth,
        )
        if not samples:
            logger.error("No samples found matching criteria")
            raise ValueError("No samples found matching criteria")

        write_sample_csv(
            output_path=output_csv,
            nodes=samples,
            session=session,
            snapshot_id=snapshot_id,
            use_heuristic=settings.use_heuristic,
            heuristic_taxonomy=settings.heuristic_taxonomy,
        )

        return len(samples)


###############################
# Ingest labeled samples
###############################


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
    unlabeled: list[int] = [i + 2 for i, row in enumerate(rows) if not row["label"]]

    if unlabeled:
        if len(unlabeled) <= 10:
            rows_str = ", ".join(map(str, unlabeled))  # ty:ignore[invalid-argument-type]
        else:
            rows_str = (
                ", ".join(map(str, unlabeled[:10])) + f", ... ({len(unlabeled)} total)"
            )

        raise ValueError(
            f"Found {len(unlabeled)} rows with missing labels.\n"
            f"Rows: {rows_str}\n"
            f"Please fill in the 'label' column for all rows."
        )


def validate_label_values(rows: List[Dict[str, Any]], valid_labels: set[str]) -> None:
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


def validate_input_csv(input_csv: Path, taxonomy: str) -> List[Dict[str, Any]]:
    """Read and validate the classification CSV."""
    try:
        rows = read_classification_csv(input_csv)
        valid_labels = get_labels(taxonomy)
        validate_all_labels_present(rows)
        validate_label_values(rows, valid_labels)
        logger.info("All labels are valid")
        return rows
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise


def create_samples_for_runs(
    index_session: Session,
    training_session: Session,
    label_runs: Dict[int, LabelRun],
) -> None:
    """Run feature extraction to create TrainingSample stubs for each label run."""
    from fine_tuning.services.feature_extraction import FeatureExtractionConfigSettings

    app_config = get_config()
    # Create a FeatureExtractionConfigSettings object with default values
    extraction_config = FeatureExtractionConfigSettings(
        batch_size=1000,
        child_cap=5,
        sibling_cap=5,
        ext_cap=10,
    )
    for snapshot_id, label_run in label_runs.items():
        num_samples = extract_features_for_run(
            index_session=index_session,
            training_session=training_session,
            snapshot_id=snapshot_id,
            app_config=app_config,
            label_run=label_run,
            settings=extraction_config,
        )
        logger.info(
            f"Created {num_samples} training samples for snapshot {snapshot_id}"
        )


def create_label_runs(session: Session, snapshot_ids: List[int]) -> Dict[int, LabelRun]:
    """Create new manual LabelRun entries for each snapshot."""
    label_runs = {}
    for snapshot_id in snapshot_ids:
        label_run = LabelRun(snapshot_id=snapshot_id, label_source="manual")
        session.add(label_run)
        label_runs[snapshot_id] = label_run
    session.flush()  # Assign IDs to label_run objects
    return label_runs


def apply_labels_to_samples(
    session: Session,
    rows: List[Dict[str, Any]],
    label_runs: Dict[int, LabelRun],
    split: Optional[str],
) -> int:
    """
    Apply labels from CSV rows to TrainingSample objects. Commits to the database
    """
    samples_by_node_id: Dict[Tuple[int, int], TrainingSample] = {}
    for snapshot_id, label_run in label_runs.items():
        samples_for_run = (
            session.query(TrainingSample).filter_by(label_run_id=label_run.id).all()
        )
        for sample in samples_for_run:
            if sample.node_id is not None:
                samples_by_node_id[(snapshot_id, sample.node_id)] = sample

    labeled_count = 0
    for row in rows:
        snapshot_id = row["snapshot_id"]
        node_id = row["node_id"]
        sample = samples_by_node_id.get((snapshot_id, node_id))

        if sample:
            sample.label = row["label"]
            sample.label_confidence = 1.0
            if split:
                sample.split = split
            labeled_count += 1
    session.commit()
    return labeled_count


def apply_sample_classifications(
    settings: ApplyClassificationsSettings,
    manager: StorageManager,
    input_csv: Path,
    taxonomy: str,
) -> None:
    rows = validate_input_csv(input_csv, taxonomy)

    with (
        manager.get_training_session() as training_session,
        manager.get_index_session(read_only=True) as index_session,
    ):
        snapshot_ids = sorted(list(set(row["snapshot_id"] for row in rows)))
        label_runs = create_label_runs(training_session, snapshot_ids)

        with manager.get_index_session(read_only=True) as index_session:
            create_samples_for_runs(index_session, training_session, label_runs)

        labeled_count = apply_labels_to_samples(
            training_session, rows, label_runs, settings.split
        )
        training_session.commit()
        logger.info(f"Applied {labeled_count} labels to training database")
