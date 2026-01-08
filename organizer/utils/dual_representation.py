"""
Utilities for building dual representation data structures.

This module contains functions to transform database models (Nodes and Categories)
into the DualRepresentation format for the v2 API.
"""

import logging
from typing import Dict, List

from api.models import DualRepresentation, Hierarchy, HierarchyItem
from data_models.pipeline import PipelineStage
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from storage.index_models import Node
from storage.manager import NodeKind, StorageManager
from storage.work_models import (
    FileMapping,
    FolderStructure,
    GroupCategory,
    GroupCategoryEntry,
    GroupIteration,
)

logger = logging.getLogger(__name__)


def dual_representation_to_folder_structure(
    dual_rep: DualRepresentation, stage_name: str
) -> dict:
    """
    Convert DualRepresentation to FolderV2-compatible structure for a specific stage.

    This provides backward compatibility for code expecting the old FolderV2 format.

    Args:
        dual_rep: The dual representation to convert
        stage_name: The stage name to convert (e.g., "original", "organized")

    Returns:
        Dict compatible with FolderV2 structure (can be parsed with FolderV2.model_validate)

    TODO: This is a compatibility layer. Consider migrating all consumers to use DualRepresentation directly.
    """
    if stage_name not in dual_rep.hierarchies:
        raise ValueError(f"Stage '{stage_name}' not found in dual representation")

    hierarchy = dual_rep.hierarchies[stage_name]

    def build_tree(item_id: str) -> dict:
        """Recursively build FolderV2-style tree from flat representation."""
        item = dual_rep.items[item_id]
        children_ids = hierarchy.tree.get(item_id, [])

        result = {
            "id": item.id,
            "name": item.name,
            "type": "folder" if item.type in ["node", "category"] else "file",
            "confidence": item.confidence,
            "possibleClassifications": item.possibleClassifications,
            "count": item.count,
        }

        if item.originalPath:
            result["originalPath"] = item.originalPath
        if item.newPath:
            result["newPath"] = item.newPath

        if children_ids:
            result["children"] = [build_tree(child_id) for child_id in children_ids]
        else:
            result["children"] = []

        return result

    return build_tree(hierarchy.root_id)


def build_dual_representation(
    storage_manager: StorageManager,
    snapshot_id: int,
    run_id: int | None = None,
    stages: List[PipelineStage] | None = None,
    save_to_db: bool = False,
) -> DualRepresentation:
    """
    Build a DualRepresentation from the database for one or more pipeline stages.

    Args:
        storage_manager: Storage manager for database access
        snapshot_id: Snapshot ID to build representation for
        run_id: Optional run ID for category/organized data
        stages: List of pipeline stages to include (defaults to [original, organized])
        save_to_db: Whether to save structures to the database

    Returns:
        DualRepresentation with items and hierarchies for requested stages
    """
    if stages is None:
        stages = [PipelineStage.original, PipelineStage.organized]

    items: Dict[str, HierarchyItem] = {}
    hierarchies: Dict[str, Hierarchy] = {}

    # Build each requested stage
    for stage in stages:
        if stage == PipelineStage.original:
            _build_original_stage(storage_manager, snapshot_id, items, hierarchies)
        elif stage == PipelineStage.organized and run_id:
            _build_organized_stage(
                storage_manager, snapshot_id, run_id, items, hierarchies, save_to_db
            )
        elif stage == PipelineStage.grouped and run_id:
            _build_grouped_stage(
                storage_manager, snapshot_id, run_id, items, hierarchies, save_to_db
            )

    return DualRepresentation(items=items, hierarchies=hierarchies)


def _build_original_stage(
    storage_manager: StorageManager,
    snapshot_id: int,
    items: Dict[str, HierarchyItem],
    hierarchies: Dict[str, Hierarchy],
) -> None:
    """Build the original filesystem hierarchy from nodes."""
    with storage_manager.get_index_session(read_only=True) as index_session:
        tree: Dict[str, List[str]] = {}
        _build_node_items_and_tree(index_session, snapshot_id, items, tree)

        # Create hierarchy object for original stage
        root_id = "original-root"
        hierarchies["original"] = Hierarchy(
            stage="original",
            source_type="node",
            tree=tree,
            root_id=root_id,
        )


def _build_organized_stage(
    storage_manager: StorageManager,
    snapshot_id: int,
    run_id: int,
    items: Dict[str, HierarchyItem],
    hierarchies: Dict[str, Hierarchy],
    save_to_db: bool = False,
) -> None:
    """Build the organized/categorized hierarchy."""
    with storage_manager.get_work_session() as work_session:
        tree: Dict[str, List[str]] = {}
        _build_category_items_and_tree(work_session, run_id, items, tree)

        # Create hierarchy object for organized stage
        root_id = "organized-root"
        hierarchies["organized"] = Hierarchy(
            stage="organized",
            source_type="category",
            tree=tree,
            root_id=root_id,
        )

        # TODO: Save to FolderStructure if save_to_db is True


def _build_grouped_stage(
    storage_manager: StorageManager,
    snapshot_id: int,
    run_id: int,
    items: Dict[str, HierarchyItem],
    hierarchies: Dict[str, Hierarchy],
    save_to_db: bool = False,
) -> None:
    """Build the grouped hierarchy (intermediate categorization stage)."""
    # TODO: Implement grouped stage building
    # This would be similar to organized but potentially different categorization
    logger.warning("Grouped stage building not yet implemented")


def _build_node_items_and_tree(
    index_session: Session,
    snapshot_id: int,
    items: Dict[str, HierarchyItem],
    tree: Dict[str, List[str]],
) -> None:
    """
    Build node items and tree structure from index database.

    Args:
        index_session: Database session for index.db
        snapshot_id: Snapshot ID to query
        items: Dictionary to populate with HierarchyItem objects
        tree: Dictionary to populate with parent-child relationships
    """
    # Query all nodes for the snapshot
    query = select(Node).where(Node.snapshot_id == snapshot_id)
    nodes = list(index_session.execute(query).scalars().all())

    logger.info(f"Building node items and tree with {len(nodes)} nodes")

    # Create a root node for the hierarchy
    root_id = "original-root"
    items[root_id] = HierarchyItem(
        id=root_id,
        name="root",
        type="node",
        originalPath=".",
    )
    tree[root_id] = []

    # Build items and tree
    for node in nodes:
        node_id = f"node-{node.id}"

        # Add to items store (only if not already present from another stage)
        if node_id not in items:
            items[node_id] = HierarchyItem(
                id=node_id,
                name=node.name,
                type="node",
                originalPath=node.abs_path,
            )

        # Build tree relationships
        if node.parent_node_id is None:
            # Top-level node, add to root
            tree[root_id].append(node_id)
        else:
            # Add to parent's children
            parent_id = f"node-{node.parent_node_id}"
            if parent_id not in tree:
                tree[parent_id] = []
            tree[parent_id].append(node_id)

        # Initialize empty children list for directories and ZIP files
        if node.kind == NodeKind.DIR or (
            node.kind == NodeKind.FILE and node.num_folder_children > 0
        ):
            if node_id not in tree:
                tree[node_id] = []

    # Second pass: populate count field for all parent nodes
    for node_id, children in tree.items():
        if node_id in items:
            items[node_id].count = len(children)


def _build_category_items_and_tree(
    work_session: Session,
    run_id: int,
    items: Dict[str, HierarchyItem],
    tree: Dict[str, List[str]],
) -> None:
    """
    Build category items and tree structure from work database.

    Args:
        work_session: Database session for work.db
        run_id: Run ID to query categories for
        items: Dictionary to populate with HierarchyItem objects (categories)
        tree: Dictionary to populate with category relationships
    """
    # Get the latest iteration for this run
    iteration_id = work_session.execute(
        select(func.max(GroupIteration.id)).where(GroupIteration.run_id == run_id)
    ).scalar()

    if iteration_id is None:
        logger.info(f"No group iterations found for run_id: {run_id}")
        return

    root_id = "organized-root"
    items[root_id] = HierarchyItem(
        id=root_id,
        name="Organized",
        type="category",
    )
    tree[root_id] = []

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

        # Add to items store with confidence from database
        items[category_id] = HierarchyItem(
            id=category_id,
            name=category.name,
            type="category",
            confidence=category.group_confidence if category.group_confidence else 1.0,
        )

        # Add to root's children
        tree[root_id].append(category_id)

        # Initialize empty children list
        tree[category_id] = []

    # Query all category entries to map files to categories and get file mappings
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

    # Get file mappings for newPath
    from storage.work_models import FileMapping

    file_mappings = (
        work_session.execute(
            select(FileMapping).where(FileMapping.run_id == run_id)
        )
        .scalars()
        .all()
    )
    mapping_dict = {fm.node_id: fm.new_path for fm in file_mappings}

    # Map nodes to categories and update confidence
    for entry in entries:
        if entry.group_id is not None:
            category_id = f"category-{entry.group_id}"
            node_id = f"node-{entry.folder_id}"

            # Add node to category's children (node may or may not already be in items)
            if category_id in tree:
                tree[category_id].append(node_id)

                # Update/create node item with confidence and newPath
                if node_id in items:
                    # Update existing node
                    if entry.confidence:
                        items[node_id].confidence = min(
                            items[node_id].confidence, entry.confidence
                        )
                    if entry.folder_id in mapping_dict:
                        items[node_id].newPath = mapping_dict[entry.folder_id]
                else:
                    # Create node item (might not be in index yet for organized view)
                    items[node_id] = HierarchyItem(
                        id=node_id,
                        name=entry.processed_name,
                        type="node",
                        confidence=entry.confidence if entry.confidence else 1.0,
                        newPath=mapping_dict.get(entry.folder_id),
                    )

    # Second pass: populate count
    for category_id, children in tree.items():
        if category_id in items:
            items[category_id].count = len(children)
