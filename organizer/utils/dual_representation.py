"""
Utilities for building dual representation data structures.

This module contains functions to transform database models (Nodes and Categories)
into Hierarchy and DualRepresentation formats for the v2 API.

Key architectural notes:
- DualRepresentation is API-level only - internal functions work with Hierarchy
- Hierarchy is analogous to FolderV2 - a single tree structure
- HierarchyItem stores intrinsic, shared data (in ItemStore)
- HierarchyRecord stores contextual, tree-specific data (including name)
"""

import logging

from api.models import (
    DualRepresentation,
    Hierarchy,
    HierarchyItem,
    HierarchyRecord,
    ItemType,
)
from data_models.pipeline import PipelineStage
from sqlalchemy import func, select
from storage.index_models import Node
from storage.manager import StorageManager
from storage.work_models import GroupCategory, GroupCategoryEntry, GroupIteration

logger = logging.getLogger(__name__)


def create_hierarchy_from_nodes(
    snapshot_id: int,
    storage_manager: StorageManager,
) -> tuple[Hierarchy, dict[str, HierarchyItem]]:
    """
    Create a hierarchy from filesystem nodes in a snapshot.

    Args:
        snapshot_id: Snapshot ID to build hierarchy for
        storage_manager: Storage manager for database access

    Returns:
        Tuple of (Hierarchy, ItemStore dict)
    """
    with storage_manager.get_index_session(read_only=True) as session:
        # Query all nodes for the snapshot
        query = select(Node).where(Node.snapshot_id == snapshot_id)
        nodes = list(session.execute(query).scalars().all())

        logger.info(f"Building node hierarchy with {len(nodes)} nodes")

        # Build items store (intrinsic data only)
        items: dict[str, HierarchyItem] = {}
        contained_ids: set[int] = set()

        # Create root item
        root_item_id = f"snapshot-{snapshot_id}-root"
        items[root_item_id] = HierarchyItem(
            id=root_item_id,
            type=ItemType.NODE,
            originalPath=".",
        )

        # Build items for all nodes
        for node in nodes:
            node_id = f"node-{node.id}"
            contained_ids.add(node.id)

            items[node_id] = HierarchyItem(
                id=node_id,
                type=ItemType.NODE,
                originalPath=node.abs_path,
            )

        # Build tree structure (parent-child relationships with names)
        # First, create a flat mapping for efficiency
        id_to_record: dict[str, HierarchyRecord] = {}
        parent_map: dict[str, list[str]] = {}  # parent_id -> child_ids

        # Create root record
        root_record = HierarchyRecord(
            itemId=root_item_id,
            name="root",
            children=[],
        )
        id_to_record[root_item_id] = root_record
        parent_map[root_item_id] = []

        # Create records for all nodes and build parent mapping
        for node in nodes:
            node_id = f"node-{node.id}"

            record = HierarchyRecord(
                itemId=node_id,
                name=node.name,
                children=[],
            )
            id_to_record[node_id] = record

            # Determine parent
            if node.parent_node_id is None:
                parent_id = root_item_id
            else:
                parent_id = f"node-{node.parent_node_id}"

            if parent_id not in parent_map:
                parent_map[parent_id] = []
            parent_map[parent_id].append(node_id)

        # Build the tree by connecting children
        for parent_id, child_ids in parent_map.items():
            if parent_id in id_to_record:
                id_to_record[parent_id].children = [
                    id_to_record[child_id] for child_id in child_ids
                ]

        # Create hierarchy
        hierarchy = Hierarchy(
            contained_ids=contained_ids,
            structure_id=0,  # Not saved to DB yet
            run_id=None,  # No run for original stage
            stage=PipelineStage.original,
            source_type=ItemType.NODE,
            root=root_record,
        )

        return hierarchy, items


def create_hierarchy_from_categories(
    run_id: int,
    snapshot_id: int,
    stage: PipelineStage,
    storage_manager: StorageManager,
) -> tuple[Hierarchy, dict[str, HierarchyItem]]:
    """
    Create a hierarchy from categories in a run.

    Args:
        run_id: Run ID to build hierarchy for
        snapshot_id: Snapshot ID (for context)
        stage: Pipeline stage (e.g., ORGANIZED, GROUPED)
        storage_manager: Storage manager for database access

    Returns:
        Tuple of (Hierarchy, ItemStore dict)
    """
    with storage_manager.get_work_session() as session:
        # Get the latest iteration for this run
        iteration_id = session.execute(
            select(func.max(GroupIteration.id)).where(GroupIteration.run_id == run_id)
        ).scalar()

        if iteration_id is None:
            logger.info(f"No group iterations found for run_id: {run_id}")
            # Return empty hierarchy
            root_item_id = f"run-{run_id}-root"
            items = {
                root_item_id: HierarchyItem(
                    id=root_item_id,
                    type=ItemType.CATEGORY,
                )
            }
            root_record = HierarchyRecord(
                itemId=root_item_id,
                name="Organized",
                children=[],
            )
            hierarchy = Hierarchy(
                contained_ids=set(),
                structure_id=0,
                run_id=run_id,
                stage=stage,
                source_type=ItemType.CATEGORY,
                root=root_record,
            )
            return hierarchy, items

        # Query all categories for this iteration
        categories = list(
            session.execute(
                select(GroupCategory).where(GroupCategory.iteration_id == iteration_id)
            )
            .scalars()
            .all()
        )

        # Query all category entries to map files to categories
        entries = list(
            session.execute(
                select(GroupCategoryEntry).where(
                    GroupCategoryEntry.iteration_id == iteration_id
                )
            )
            .scalars()
            .all()
        )

        logger.info(
            f"Building category hierarchy with {len(categories)} categories "
            f"and {len(entries)} entries"
        )

        # Build items store
        items: dict[str, HierarchyItem] = {}
        contained_ids: set[int] = set()

        # Create root item
        root_item_id = f"run-{run_id}-root"
        items[root_item_id] = HierarchyItem(
            id=root_item_id,
            type=ItemType.CATEGORY,
        )

        # Add categories to items
        for category in categories:
            category_id = f"category-{category.id}"
            contained_ids.add(category.id)

            items[category_id] = HierarchyItem(
                id=category_id,
                type=ItemType.CATEGORY,
            )

        # Add nodes referenced by entries to items
        for entry in entries:
            if entry.group_id is not None:
                node_id = f"node-{entry.folder_id}"
                # Note: We don't add to contained_ids for nodes since they're not
                # part of the category table

                if node_id not in items:
                    items[node_id] = HierarchyItem(
                        id=node_id,
                        type=ItemType.NODE,
                    )

        # Build tree structure
        id_to_record: dict[str, HierarchyRecord] = {}
        parent_map: dict[str, list[str]] = {}

        # Create root record
        root_record = HierarchyRecord(
            itemId=root_item_id,
            name="Organized",
            children=[],
        )
        id_to_record[root_item_id] = root_record
        parent_map[root_item_id] = []

        # Create records for categories
        for category in categories:
            category_id = f"category-{category.id}"

            record = HierarchyRecord(
                itemId=category_id,
                name=category.name,
                children=[],
            )
            id_to_record[category_id] = record
            parent_map[root_item_id].append(category_id)
            parent_map[category_id] = []

        # Create records for nodes and map to categories
        for entry in entries:
            if entry.group_id is not None:
                category_id = f"category-{entry.group_id}"
                node_id = f"node-{entry.folder_id}"

                if node_id not in id_to_record:
                    # Use processed_name from entry for contextual naming
                    record = HierarchyRecord(
                        itemId=node_id,
                        name=entry.processed_name or "Unknown",
                        children=[],
                    )
                    id_to_record[node_id] = record

                # Add to parent's children list
                if category_id in parent_map:
                    parent_map[category_id].append(node_id)

        # Build the tree by connecting children
        for parent_id, child_ids in parent_map.items():
            if parent_id in id_to_record:
                id_to_record[parent_id].children = [
                    id_to_record[child_id]
                    for child_id in child_ids
                    if child_id in id_to_record
                ]

        # Create hierarchy
        hierarchy = Hierarchy(
            contained_ids=contained_ids,
            structure_id=0,  # Not saved to DB yet
            run_id=run_id,
            stage=stage,
            source_type=ItemType.CATEGORY,
            root=root_record,
        )

        return hierarchy, items


def convert_hierarchy_to_folder_structure(
    hierarchy: Hierarchy, items: dict[str, HierarchyItem]
) -> dict:
    """
    Convert a Hierarchy to FolderV2-compatible structure.

    This provides backward compatibility for code expecting the old FolderV2 format.

    Args:
        hierarchy: The hierarchy to convert
        items: The ItemStore containing shared item data

    Returns:
        Dict compatible with FolderV2 structure (can be parsed with FolderV2.model_validate)
    """

    def build_tree(record: HierarchyRecord) -> dict:
        """Recursively build FolderV2-style tree from HierarchyRecord."""
        item = items[record.itemId]

        result = {
            "id": item.id,
            "name": record.name,  # Get name from record (contextual)
            "type": "folder"
            if item.type in [ItemType.NODE, ItemType.CATEGORY]
            else "file",
            "children": [],
        }

        if item.originalPath:
            result["originalPath"] = item.originalPath

        if record.children:
            result["children"] = [build_tree(child) for child in record.children]

        return result

    return build_tree(hierarchy.root)


def build_dual_representation(
    storage_manager: StorageManager,
    snapshot_id: int,
    run_id: int | None = None,
    stages: list[PipelineStage] | None = None,
) -> DualRepresentation:
    """
    Build a DualRepresentation from the database for one or more pipeline stages.

    This is an API-level function that assembles multiple Hierarchy objects
    into a single payload. Internal functions should work with individual
    Hierarchy objects.

    Args:
        storage_manager: Storage manager for database access
        snapshot_id: Snapshot ID to build representation for
        run_id: Optional run ID for category/organized data
        stages: List of pipeline stages to include (defaults to [ORIGINAL, ORGANIZED])

    Returns:
        DualRepresentation with items and hierarchies for requested stages
    """
    if stages is None:
        stages = [PipelineStage.original, PipelineStage.organized]

    # Collect items and hierarchies
    all_items: dict[str, HierarchyItem] = {}
    hierarchies: dict[str, Hierarchy] = {}

    # Build each requested stage
    for stage in stages:
        if stage == PipelineStage.original:
            hierarchy, items = create_hierarchy_from_nodes(snapshot_id, storage_manager)
            hierarchies[stage.value] = hierarchy
            all_items.update(items)  # Merge into shared ItemStore

        elif stage in [PipelineStage.organized, PipelineStage.grouped] and run_id:
            hierarchy, items = create_hierarchy_from_categories(
                run_id, snapshot_id, stage, storage_manager
            )
            hierarchies[stage.value] = hierarchy
            all_items.update(items)  # Merge into shared ItemStore

    return DualRepresentation(items=all_items, hierarchies=hierarchies)
