from collections import defaultdict
from logging import getLogger
from typing import Dict, List

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from storage.manager import NodeKind
from storage.index_models import Node
from storage.work_models import GroupCategoryEntry, GroupIteration

from stages.grouping.helpers import common_token_grouping, get_next_iteration_id
from stages.grouping.constants import STOP_WORDS

logger = getLogger(__name__)

# A confidence threshold to decide whether to accept a split
DECOMPOSITION_CONFIDENCE_THRESHOLD = 0.6


def calculate_split_confidence(part1: str, part2: str, original_name: str) -> float:
    """
    Calculates a confidence score for a proposed binary split of a name.
    The score is based on a set of heuristic rules.

    Args:
        part1: The first part (prefix) of the split
        part2: The second part (suffix) of the split
        original_name: The original unsplit name

    Returns:
        A confidence score between 0.0 and 1.0
    """
    score = 1.0

    # Rule: First part should generally be longer than the second
    if len(part1) <= len(part2):
        # Apply a penalty, stronger if the first part is much shorter
        score *= 0.8 * (len(part1) / (len(part2) + 1e-6))

    # Rule: The last section should not be more than 50% of the original length
    if len(part2) > 0.5 * len(original_name):
        score *= 0.7

    # Rule: Penalize groups that are mostly stopwords
    part1_words = [w for w in part1.lower().split() if w not in STOP_WORDS]
    part2_words = [w for w in part2.lower().split() if w not in STOP_WORDS]

    # The prefix (part1) must contain non-stopwords
    if not part1_words:
        return 0.0  # A prefix of only stopwords is invalid

    # Penalize if the second part has more meaningful content than the first
    if len(" ".join(part1_words)) < len(" ".join(part2_words)):
        score *= 0.9

    return score


def decompose_name(name: str) -> List[str]:
    """
    Recursively decomposes a name into its best constituent parts based on a confidence score.

    This function implements hierarchical name decomposition by repeatedly finding
    the highest-confidence binary split point in a name.

    Args:
        name: The name to decompose

    Returns:
        A list of strings representing the decomposed parts
    """
    words = name.split()
    if len(words) <= 1:
        return [name]

    best_split = None
    max_confidence = -1.0

    # Find the best binary split for the current name
    for i in range(1, len(words)):
        part1 = " ".join(words[:i])
        part2 = " ".join(words[i:])

        confidence = calculate_split_confidence(part1, part2, name)

        if confidence > max_confidence:
            max_confidence = confidence
            best_split = (part1, part2)

    # If the best split is good enough, recurse on the second part
    if max_confidence > DECOMPOSITION_CONFIDENCE_THRESHOLD and best_split:
        # Return the first part, plus the result of decomposing the second part
        return [best_split[0]] + decompose_name(best_split[1])
    else:
        # If no split is confident enough, do not decompose the name further
        return [name]


def _get_folder_groups(session: Session) -> Dict[int, List[Node]]:
    """
    Retrieve folders grouped by parent node where there is more than one folder
    in the group.

    Returns:
        Dict[int, List[Node]]: Dictionary with parent node IDs as keys and lists of folders as values
    """
    # First find parent nodes that have multiple folder children
    parent_counts = (
        session.query(Node.parent_node_id, func.count(Node.id).label("folder_count"))
        .filter(Node.kind == NodeKind.DIR)
        .group_by(Node.parent_node_id)
        .having(func.count(Node.id) > 1)
        .subquery()
    )

    # Then get all folders that belong to these parent nodes
    folders = (
        session.query(Node)
        .filter(Node.kind == NodeKind.DIR)
        .join(parent_counts, Node.parent_node_id == parent_counts.c.parent_node_id)
        .order_by(Node.parent_node_id, Node.name)
        .all()
    )

    # Group the folders by parent node ID
    grouped_folders = {}
    for folder in folders:
        parent_id = folder.parent_node_id if folder.parent_node_id else 0
        grouped_folders.setdefault(parent_id, []).append(folder)

    return grouped_folders


def process_folders(session: Session):
    """
    Process groups of folders and update the classification in the database.

    NOTE: This function has been updated to work with the new storage system.
    However, Node is immutable and doesn't have a 'categories' field.
    Categories should be stored in the work database if this functionality is needed.
    This function now only returns the categorization without storing it.

    Args:
        session (Session): SQLAlchemy session for index database

    Returns:
        Dict mapping node_id to categories
    """

    grouped_folders = _get_folder_groups(session)
    node_categories = {}

    for _, folders in grouped_folders.items():
        # Use normalized_name from features if available, otherwise use clean_filename
        from utils.filename_processing import clean_filename

        cleaned_name_to_id = {
            clean_filename(folder.name): folder.id for folder in folders
        }
        token_grouping = common_token_grouping(list(cleaned_name_to_id.keys()))
        if token_grouping:
            for name, category in token_grouping.items():
                node_id = cleaned_name_to_id[name]
                node_categories[node_id] = category

    # Note: Can't commit changes to Node as it's immutable
    # Categories would need to be stored in work database
    return node_categories


def _is_valid_group_prefix(prefix: str) -> bool:
    """
    Check if a prefix is valid for grouping.
    Returns False if the prefix consists only of stopwords.
    Returns True if the prefix contains at least one non-stopword.
    """
    if not prefix or not prefix.strip():
        return False

    # Split prefix into words and check if at least one is not a stopword
    words = prefix.lower().split()
    non_stopwords = [word for word in words if word not in STOP_WORDS]

    return len(non_stopwords) > 0


def apply_hierarchical_decomposition(
    session: Session,
    run_id: int,
    snapshot_id: int,
) -> None:
    """
    Apply hierarchical name decomposition to break down folder names into meaningful parts.

    This is a more flexible, rule-based decomposition that is applied to each name
    individually. Similar folder names will be decomposed into similar sets of parts,
    achieving grouping as an emergent property.
    """
    logger.info("Applying hierarchical folder name decomposition")

    # Get the current iteration ID and create a new iteration
    iteration_id = get_next_iteration_id(session)

    iteration = GroupIteration(
        id=iteration_id,
        run_id=run_id,
        snapshot_id=snapshot_id,
        description="Hierarchical name decomposition",
    )
    session.add(iteration)
    session.flush()

    # Get entries from the previous iteration
    stmt = select(GroupCategoryEntry).where(
        GroupCategoryEntry.iteration_id == iteration_id - 1
    )
    previous_entries = session.scalars(stmt).all()

    if not previous_entries:
        logger.info("No entries to process")
        return

    for entry in previous_entries:
        processed_name = entry.processed_name or entry.pre_processed_name
        if not processed_name:
            continue

        # Decompose the name into its constituent parts
        decomposed_parts = decompose_name(processed_name)

        # Create new entries for each decomposed part
        for part in decomposed_parts:
            if not part or not part.strip():
                continue

            new_entry = GroupCategoryEntry(
                folder_id=entry.folder_id,
                iteration_id=iteration_id,
                pre_processed_name=entry.pre_processed_name,
                processed_name=part,
                path=entry.path,
                confidence=entry.confidence,
                processed=False,
                derived_names=entry.derived_names,
            )
            session.add(new_entry)

    session.commit()
    logger.info("Hierarchical folder name decomposition complete")


def apply_common_prefix_grouping(
    session: Session,
    run_id: int,
    snapshot_id: int,
) -> None:
    """
    Apply folder name grouping to identify common prefixes and create hierarchical groups.

    This step processes groups from the previous iteration and uses common_token_grouping
    to identify folder names that share common prefixes across multiple names.
    """
    logger.info("Applying common prefix grouping")

    # Get the current iteration ID and create a new iteration
    iteration_id = get_next_iteration_id(session)

    iteration = GroupIteration(
        id=iteration_id,
        run_id=run_id,
        snapshot_id=snapshot_id,
        description="Folder name prefix grouping",
    )
    session.add(iteration)
    session.flush()

    # Get entries from the previous iteration
    stmt = select(GroupCategoryEntry).where(
        GroupCategoryEntry.iteration_id == iteration_id - 1
    )
    previous_entries = session.scalars(stmt).all()

    if not previous_entries:
        logger.info("No entries to process")
        return

    # Create a mapping of processed_name to list of entries
    name_to_entries = defaultdict(list)
    for entry in previous_entries:
        processed_name = entry.processed_name or entry.pre_processed_name
        if processed_name:
            name_to_entries[processed_name].append(entry)

    # Get the list of unique processed names
    names_to_process = list(name_to_entries.keys())

    # Apply common_token_grouping to identify shared prefixes
    grouping_result = common_token_grouping(names_to_process)

    # Track which entries have been processed
    processed_names = set()

    if grouping_result:
        # Process grouped entries
        for original_name, components in grouping_result.items():
            if len(components) < 2:
                continue

            prefix = components[0]
            suffix = components[1] if len(components) > 1 else None

            # Validate that the prefix is not stopword-only
            if not _is_valid_group_prefix(prefix):
                logger.debug(f"Skipping stopword-only prefix: {prefix}")
                continue

            # For each entry with this processed_name, create new entries
            for entry in name_to_entries[original_name]:
                # Create entry for the common prefix
                prefix_entry = GroupCategoryEntry(
                    folder_id=entry.folder_id,
                    iteration_id=iteration_id,
                    pre_processed_name=entry.pre_processed_name,
                    processed_name=prefix,
                    path=entry.path,
                    confidence=entry.confidence,
                    processed=False,
                    derived_names=entry.derived_names,
                )
                session.add(prefix_entry)

                # Create entry for the suffix (if it exists and is valid)
                if suffix:
                    suffix_entry = GroupCategoryEntry(
                        folder_id=entry.folder_id,
                        iteration_id=iteration_id,
                        pre_processed_name=entry.pre_processed_name,
                        processed_name=suffix,
                        path=entry.path,
                        confidence=entry.confidence,
                        processed=False,
                        derived_names=entry.derived_names,
                    )
                    session.add(suffix_entry)

            processed_names.add(original_name)

    # Copy over ungrouped entries
    for name, entries in name_to_entries.items():
        if name not in processed_names:
            for entry in entries:
                new_entry = GroupCategoryEntry(
                    folder_id=entry.folder_id,
                    iteration_id=iteration_id,
                    pre_processed_name=entry.pre_processed_name,
                    processed_name=entry.processed_name,
                    path=entry.path,
                    confidence=entry.confidence,
                    processed=False,
                    derived_names=entry.derived_names,
                )
                session.add(new_entry)

    session.commit()
    logger.info("Common prefix grouping complete")
