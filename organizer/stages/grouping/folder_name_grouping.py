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


def apply_folder_name_grouping(
    session: Session,
    run_id: int,
    snapshot_id: int,
) -> None:
    """
    Apply folder name grouping to identify common prefixes and create hierarchical groups.
    This step processes groups from the previous iteration and uses common_token_grouping
    to identify folder names that share common prefixes.
    """
    logger.info("Applying folder name grouping")

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
    grouping_result = common_token_grouping(names_to_process, prefer_longer_names=True)

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
    logger.info("Folder name grouping complete")
