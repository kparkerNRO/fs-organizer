import json
from logging import getLogger
import os
from pathlib import Path
from typing import cast

from data_models.pipeline import File, FolderV2, PipelineStage
from sqlalchemy import func, select
from sqlalchemy import select as sql_select
from sqlalchemy.orm import Session
from storage.index_models import Node
from storage.id_defaults import get_latest_run
from storage.index_models import Snapshot
from storage.manager import NodeKind, StorageManager
from storage.work_models import FileMapping, FolderStructure, GroupCategoryEntry

logger = getLogger(__name__)


def get_newest_entry_for_stage(
    work_session: Session, structure_type: PipelineStage, run_id: int | None
):
    if not run_id and structure_type != PipelineStage.original:
        run = get_latest_run(work_session)
        if not run:
            return None
        run_id = run.id

    logger.info(
        f"Getting structure definition for type {structure_type} with run_id {run_id}"
    )

    query = select(FolderStructure).where(
        FolderStructure.structure_type == structure_type
    )

    if run_id:
        query = query.where(FolderStructure.run_id == run_id)

    newest_entry = work_session.execute(
        query.order_by(FolderStructure.id.desc()).limit(1)
    ).scalar_one_or_none()

    if newest_entry:
        entry = newest_entry.structure
        return entry
    return None


def get_folder_heirarchy(
    manager: StorageManager, run_id: int, structure_type: PipelineStage
):
    logger.debug(f"Current working directory: {os.getcwd()}")

    with manager.get_work_session() as session:
        entry = get_newest_entry_for_stage(session, structure_type, run_id)
        if entry is not None:
            entry = json.loads(entry)
            pretty_entry = json.dumps(entry, indent=4)
            logger.info(pretty_entry)
        return entry


def sort_folder_structure(folder_data: dict) -> dict:
    """
    Recursively sort folder structure by folder/file names
    """
    if not isinstance(folder_data, dict):
        return folder_data

    # Create a new FolderV2 object to ensure proper structure
    if "name" in folder_data and "children" in folder_data:
        # This is a folder object
        sorted_children = []

        # Sort children by name
        children = folder_data.get("children", [])
        if children:
            # Separate files and folders by type field
            files = [child for child in children if child.get("type") == "file"]
            folders = [child for child in children if child.get("type") == "folder"]

            # Sort files by name
            files.sort(key=lambda x: x.get("name", "").lower())

            # Sort folders by name and recursively sort their children
            folders.sort(key=lambda x: x.get("name", "").lower())
            for folder in folders:
                sorted_children.append(sort_folder_structure(folder))

            # Add files after folders
            sorted_children.extend(files)

        # Return sorted folder
        return {**folder_data, "children": sorted_children}

    return folder_data


def _build_tree_structure(
    nodes: list[Node],
) -> tuple[int, FolderV2]:
    """
    Build hierarchical tree structure from flat list of nodes.

    Args:
        nodes: Flat list of Node objects

    Returns:
        count, and FolderV2 with the tree structure
    """
    # Create children map
    children_map = {}
    root_nodes = []

    for node in nodes:
        if node.parent_node_id is None:
            root_nodes.append(node)
        else:
            if node.parent_node_id not in children_map:
                children_map[node.parent_node_id] = []
            children_map[node.parent_node_id].append(node)

    def node_to_folder(node: Node) -> File | FolderV2:
        """Convert a Node to a dictionary representation."""
        # Check if node has children (e.g., ZIP files are FILE kind but have children)
        has_children = node.id in children_map

        if node.kind == NodeKind.FILE and not has_children:
            return File.from_node(node)
        else:
            # Treat as folder if it's a directory OR if it's a file with children (like ZIP)
            folder = FolderV2.from_node(node)

            # Add children recursively
            if has_children:
                folder_children = [
                    node_to_folder(child) for child in children_map[node.id]
                ]

                folder.children = folder_children
                folder.count = len(children_map[node.id])
            return folder

    # Build complete tree structure
    if not root_nodes:
        raise ValueError(
            f"No root nodes found. Total nodes: {len(nodes)}. All nodes have a parent_node_id set."
        )

    structure = [
        node_to_folder(root) for root in sorted(root_nodes, key=lambda n: n.name)
    ]

    structure_root = FolderV2(
        id=0,
        name="root",
        originalPath=".",
        children=structure,
        count=len(nodes),
    )

    return len(nodes), structure_root


def create_folder_structure_for_snapshot(
    index_session: Session,
    include_files: bool = False,
    *,
    snapshot_id: int | None = None,
    associated_run: int | None = None,
) -> FolderStructure:
    # Get most recent snapshot
    query = sql_select(Snapshot)
    if snapshot_id:
        query = query.where(Snapshot.id == snapshot_id)
    else:
        query = query.order_by(Snapshot.created_at.desc())

    snapshot = index_session.execute(query).scalars().first()

    if not snapshot:
        raise ValueError("No snapshots found in database")

    # Query nodes, optionally filtering files
    query = sql_select(Node).where(Node.snapshot_id == snapshot.id)
    if not include_files:
        query = query.where(Node.kind == NodeKind.DIR)

    nodes = list(index_session.execute(query.order_by(Node.rel_path)).scalars().all())

    # Build tree structure
    count, structure = _build_tree_structure(nodes)

    processed_structure = FolderStructure(
        snapshot_id=snapshot.id,
        run_id=associated_run,
        structure_type=PipelineStage.original,
        structure=structure.model_dump(),
        total_nodes=count,
    )
    return processed_structure


def export_snapshot_structure(
    output_path: Path,
    storage: StorageManager,
    include_files: bool = False,
) -> dict:
    """
    Export the most recent snapshot's directory structure to a JSON file.

    Args:
        output_path: Path where the JSON file will be written
        storage_path: Storage directory containing index.db (None for default)
        include_files: Whether to include files in the structure (default: directories only)

    Returns:
        Dictionary containing the exported structure metadata

    Raises:
        ValueError: If no snapshots are found in the database
    """

    with storage.get_index_session(read_only=True) as index_session:
        file_structure = create_folder_structure_for_snapshot(
            index_session, include_files
        )
        # Write to JSON file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # TODO: investigate using the sqlalchemy/pydantic model for this
        converted_structure = {
            "run_id": file_structure.run_id,
            "snapshot_id": file_structure.snapshot_id,
            "total_nodes": file_structure.total_nodes,
            "structure_type": file_structure.structure_type,
            "structure": file_structure.structure,
        }

        with open(output_path, "w") as f:
            json.dump(converted_structure, f, indent=2)

        return {
            "snapshot_id": file_structure.snapshot_id,
            "created_at": file_structure.created_at,
            "total_nodes": file_structure.total_nodes,
            "output_path": str(output_path),
        }


def _insert_file_in_structure(
    folder_structure: FolderV2,
    file: Node,
    parts: list[tuple[str | None, int | float]],
    new_path: str | None = None,
):
    current_representation = folder_structure

    for component in parts:
        if isinstance(component, tuple):
            component, confidence = component
        else:
            confidence = 1
        if component not in current_representation.children_map.keys():
            current_representation.children.append(
                FolderV2(name=component, confidence=confidence)
            )
        current_representation = current_representation.children_map[component]

    current_representation.children.append(
        File(
            id=file.id,
            name=file.name,
            originalPath=file.abs_path,
            newPath=new_path,
        )
    )


def get_groups_for_node(
    index_session: Session,
    work_session: Session,
    node: Node,
    iteration_id: int,
) -> list[GroupCategoryEntry]:
    """
    recursively get categories for the provided path
    """
    parent = index_session.execute(
        select(Node).where(Node.id == node.parent_node_id)
    ).scalar_one_or_none()

    if not parent:
        return []

    # GroupCategoryEntry uses folder_id which maps to node_id in the new schema
    groups = (
        work_session.execute(
            select(GroupCategoryEntry)
            .where(GroupCategoryEntry.iteration_id == iteration_id)
            .where(GroupCategoryEntry.folder_id == parent.id)
        )
        .scalars()
        .all()
    )

    categories = get_groups_for_node(index_session, work_session, parent, iteration_id)
    category_names = {cat.processed_name: index for index, cat in enumerate(categories)}
    for group in groups:
        processed_name = group.processed_name
        if group.processed_name in category_names:
            categories[category_names[processed_name]].confidence = min(
                group.confidence, categories[category_names[processed_name]].confidence
            )
        else:
            categories.append(group)

    return categories


def calculate_folder_structure_for_stage(
    manager: StorageManager,
    snapshot_id: int,
    run_id: int,
    structure_type: PipelineStage = PipelineStage.organized,
    # category_resolver: Callable[
    #     [Session, Session, str | Path, int | None], list[GroupCategoryEntry]
    # ] = get_categories_for_path,
):
    with (
        manager.get_index_session(read_only=True) as index_session,
        manager.get_work_session() as work_session,
    ):
        # Get all file nodes from index database
        files = (
            index_session.execute(
                select(Node)
                .where(Node.snapshot_id == snapshot_id, Node.kind == NodeKind.FILE)
                .limit(5000)
            )
            .scalars()
            .all()
        )

        # Get latest iteration_id from work database
        iteration_id = work_session.execute(
            select(func.max(GroupCategoryEntry.iteration_id))
        ).scalar_one()

        if iteration_id is None:
            logger.warning("No group categories available for categorization.")
            return None

        total_files = len(files)
        logger.info(f"Processing {total_files} files...")

        # Process each file
        folder_structure = FolderV2(name="Root")
        for i, file in enumerate(files, 1):
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{total_files} files")

            # Get categories using index session for node lookups and work session for groups
            categories = get_groups_for_node(
                index_session,
                work_session,
                file,
                iteration_id,
            )
            names = [cast(str, cat.processed_name) for cat in categories]
            new_path = "/".join(names)

            # Update or create FileMapping in work database
            existing_mapping = work_session.execute(
                select(FileMapping).where(
                    FileMapping.run_id == run_id, FileMapping.node_id == file.id
                )
            ).scalar_one_or_none()

            if existing_mapping:
                existing_mapping.new_path = new_path
            else:
                work_session.add(
                    FileMapping(
                        run_id=run_id,
                        node_id=file.id,
                        original_path=file.abs_path,
                        new_path=new_path,
                    )
                )

            category_names = [
                (category.processed_name, category.confidence)
                for category in categories
            ]
            _insert_file_in_structure(folder_structure, file, category_names, new_path)

        work_session.add(
            FolderStructure(
                run_id=run_id,
                snapshot_id=snapshot_id,
                total_nodes=total_files,
                structure_type=structure_type.value,
                structure=folder_structure.model_dump(),
            )
        )
        work_session.commit()
