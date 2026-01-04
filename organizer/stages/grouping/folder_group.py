from typing import Dict, List

from sqlalchemy import func
from sqlalchemy.orm import Session
from storage.index_models import Node

from stages.grouping.helpers import common_token_grouping


def get_folder_groups(session: Session) -> Dict[int, List[Node]]:
    """
    Retrieve folders grouped by parent node where there is more than one folder
    in the group.

    Returns:
        Dict[int, List[Node]]: Dictionary with parent node IDs as keys and lists of folders as values
    """
    # First find parent nodes that have multiple folder children
    parent_counts = (
        session.query(Node.parent_node_id, func.count(Node.id).label("folder_count"))
        .filter(Node.kind == "dir")
        .group_by(Node.parent_node_id)
        .having(func.count(Node.id) > 1)
        .subquery()
    )

    # Then get all folders that belong to these parent nodes
    folders = (
        session.query(Node)
        .filter(Node.kind == "dir")
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

    grouped_folders = get_folder_groups(session)
    node_categories = {}

    for _, folders in grouped_folders.items():
        # Use normalized_name from features if available, otherwise use clean_filename
        from utils.filename_processing import clean_filename

        cleaned_name_to_id = {clean_filename(folder.name): folder.id for folder in folders}
        token_grouping = common_token_grouping(list(cleaned_name_to_id.keys()))
        if token_grouping:
            for name, category in token_grouping.items():
                node_id = cleaned_name_to_id[name]
                node_categories[node_id] = category

    # Note: Can't commit changes to Node as it's immutable
    # Categories would need to be stored in work database
    return node_categories
