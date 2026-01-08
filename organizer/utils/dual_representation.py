"""
Utilities for building dual representation data structures.

This module contains functions to transform database models (Nodes and Categories)
into the DualRepresentation format for the v2 API.
"""

import logging
from typing import Dict, List

from api.models import DualRepresentation, HierarchyItem
from data_models.pipeline import PipelineStage
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from storage.index_models import Node
from storage.manager import NodeKind, StorageManager
from storage.work_models import FolderStructure, GroupCategory, GroupCategoryEntry, GroupIteration

logger = logging.getLogger(__name__)


def build_dual_representation(
    storage_manager: StorageManager,
    snapshot_id: int,
    run_id: int | None = None,
    structure_type: PipelineStage = PipelineStage.organized,
    save_to_db: bool = False,
) -> DualRepresentation:
    """
    Build a DualRepresentation from the database.

    Args:
        storage_manager: Storage manager for database access
        snapshot_id: Snapshot ID to build representation for
        run_id: Optional run ID for category data
        structure_type: Type of structure (original, organized, etc.)
        save_to_db: Whether to save the structure to the database

    Returns:
        DualRepresentation with items and both hierarchies
    """
    items: Dict[str, HierarchyItem] = {}
    node_hierarchy: Dict[str, List[str]] = {}
    category_hierarchy: Dict[str, List[str]] = {}

    with storage_manager.get_index_session(read_only=True) as index_session:
        # Build node hierarchy from index.db
        _build_node_hierarchy(index_session, snapshot_id, items, node_hierarchy)

    # Always create category root, even if there are no categories
    root_id = "category-root"
    items[root_id] = HierarchyItem(
        id=root_id,
        name="Categories",
        type="category",
    )
    category_hierarchy[root_id] = []

    if run_id:
        with storage_manager.get_work_session() as work_session:
            # Build category hierarchy from work.db
            _build_category_hierarchy(work_session, run_id, items, category_hierarchy)

            # Save to database if requested
            if save_to_db:
                dual_rep = DualRepresentation(
                    items=items,
                    node_hierarchy=node_hierarchy,
                    category_hierarchy=category_hierarchy,
                )

                # TODO: Determine total node count from items
                total_nodes = len([item for item in items.values() if item.type == "node"])

                folder_structure = FolderStructure(
                    run_id=run_id,
                    snapshot_id=snapshot_id,
                    structure_type=structure_type.value,
                    structure=dual_rep.model_dump(),
                    total_nodes=total_nodes,
                )
                work_session.add(folder_structure)
                work_session.commit()

    return DualRepresentation(
        items=items,
        node_hierarchy=node_hierarchy,
        category_hierarchy=category_hierarchy,
    )


def _build_node_hierarchy(
    index_session: Session,
    snapshot_id: int,
    items: Dict[str, HierarchyItem],
    node_hierarchy: Dict[str, List[str]],
) -> None:
    """
    Build the node hierarchy from the index database.

    Args:
        index_session: Database session for index.db
        snapshot_id: Snapshot ID to query
        items: Dictionary to populate with HierarchyItem objects
        node_hierarchy: Dictionary to populate with parent-child relationships
    """
    # Query all nodes for the snapshot
    query = select(Node).where(Node.snapshot_id == snapshot_id)
    nodes = list(index_session.execute(query).scalars().all())

    logger.info(f"Building node hierarchy with {len(nodes)} nodes")

    # Create a root node for the hierarchy
    root_id = "node-root"
    items[root_id] = HierarchyItem(
        id=root_id,
        name="root",
        type="node",
        originalPath=".",
    )
    node_hierarchy[root_id] = []

    # Build items and hierarchy
    for node in nodes:
        node_id = f"node-{node.id}"

        # Add to items store
        items[node_id] = HierarchyItem(
            id=node_id,
            name=node.name,
            type="node",
            originalPath=node.abs_path,
        )

        # Build hierarchy relationships
        if node.parent_node_id is None:
            # Top-level node, add to root
            node_hierarchy[root_id].append(node_id)
        else:
            # Add to parent's children
            parent_id = f"node-{node.parent_node_id}"
            if parent_id not in node_hierarchy:
                node_hierarchy[parent_id] = []
            node_hierarchy[parent_id].append(node_id)

        # Initialize empty children list for directories and ZIP files
        if node.kind == NodeKind.DIR or (
            node.kind == NodeKind.FILE and node.num_folder_children > 0
        ):
            if node_id not in node_hierarchy:
                node_hierarchy[node_id] = []


def _build_category_hierarchy(
    work_session: Session,
    run_id: int,
    items: Dict[str, HierarchyItem],
    category_hierarchy: Dict[str, List[str]],
) -> None:
    """
    Build the category hierarchy from the work database.

    Args:
        work_session: Database session for work.db
        run_id: Run ID to query categories for
        items: Dictionary to populate with HierarchyItem objects (categories)
        category_hierarchy: Dictionary to populate with category relationships
    """
    # Get the latest iteration for this run
    iteration_id = work_session.execute(
        select(func.max(GroupIteration.id)).where(GroupIteration.run_id == run_id)
    ).scalar()

    if iteration_id is None:
        logger.info(f"No group iterations found for run_id: {run_id}")
        return

    root_id = "category-root"

    # Query all categories for this iteration
    categories = (
        work_session.execute(
            select(GroupCategory).where(GroupCategory.iteration_id == iteration_id)
        )
        .scalars()
        .all()
    )

    logger.info(f"Building category hierarchy with {len(categories)} categories")

    # Add categories to items and hierarchy
    for category in categories:
        category_id = f"category-{category.id}"

        # Add to items store
        items[category_id] = HierarchyItem(
            id=category_id,
            name=category.name,
            type="category",
        )

        # Add to root's children
        category_hierarchy[root_id].append(category_id)

        # Initialize empty children list
        category_hierarchy[category_id] = []

    # Query all category entries to map files to categories
    entries = (
        work_session.execute(
            select(GroupCategoryEntry).where(
                GroupCategoryEntry.iteration_id == iteration_id
            )
        )
        .scalars()
        .all()
    )

    logger.info(f"Processing {len(entries)} category entries")

    # Map nodes to categories
    for entry in entries:
        if entry.group_id is not None:
            category_id = f"category-{entry.group_id}"
            node_id = f"node-{entry.folder_id}"

            # Only add if the node exists in items (it should from node hierarchy)
            if node_id in items and category_id in category_hierarchy:
                category_hierarchy[category_id].append(node_id)
