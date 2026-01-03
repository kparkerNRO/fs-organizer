"""
Responsible for post-grouping processing

Apply categories at the file level,
combine groups that were over-zealously split
generate folder paths based on that

"""

import logging
from pathlib import Path
from typing import cast
from collections.abc import Callable

from sqlalchemy.orm import Session
from sqlalchemy import func, select

from api.api import FolderV2, StructureType
from storage.manager import StorageManager
from storage.index_models import Node
from storage.work_models import FolderStructure, GroupCategoryEntry, FileMapping

from utils.folder_structure import insert_file_in_structure

logger = logging.getLogger(__name__)


def _insert_file_in_structure(
    folder_structure: FolderV2,
    file: Node,
    parts: list[tuple[str, float]],
    new_path: str | None,
) -> None:
    insert_file_in_structure(folder_structure, file, parts, new_path)


def get_parent_folder(
    session: Session, parent_path: Path, zip_content: bool = False
) -> Node | None:
    """Find the parent folder entry based on its path."""

    parent_path_str = str(parent_path)
    parent = session.execute(
        select(Node).where(Node.abs_path == parent_path_str, Node.kind == "dir")
    ).scalar_one_or_none()

    if not parent and zip_content:
        parent_path = parent_path.parent
        parent = session.execute(
            select(Node).where(Node.abs_path == str(parent_path), Node.kind == "dir")
        ).scalar_one_or_none()

    return parent


def get_categories_for_path(
    index_session: Session,
    work_session: Session,
    path: str | Path,
    iteration_id: int | None,
) -> list[GroupCategoryEntry]:
    """
    recursively get categories for the provided path
    """
    if iteration_id is None:
        return []

    if isinstance(path, str):
        path = Path(path)

    parent_path = path.parent
    zip_content = parent_path.match("*.zip")
    parent = get_parent_folder(index_session, parent_path, zip_content)

    if not parent:
        return []

    # GroupCategoryEntry uses folder_id which maps to node_id in the new schema
    groups = (
        work_session.execute(
            select(GroupCategoryEntry)
            .where(GroupCategoryEntry.iteration_id == iteration_id)
            .where(GroupCategoryEntry.folder_id == parent.node_id)
        )
        .scalars()
        .all()
    )

    categories = get_categories_for_path(
        index_session,
        work_session,
        parent_path,
        iteration_id,
    )
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


def calculate_folder_structure(
    manager: StorageManager,
    snapshot_id: int,
    run_id: int,
    structure_type: StructureType = StructureType.organized,
    category_resolver: Callable[
        [Session, Session, str | Path, int | None], list[GroupCategoryEntry]
    ]
    | None = None,
    insert_file: Callable[
        [FolderV2, Node, list[tuple[str, float]], str | None], None
    ] = _insert_file_in_structure,
):
    with (
        manager.get_index_session(read_only=True) as index_session,
        manager.get_work_session() as work_session,
    ):
        # Get all file nodes from index database
        files = (
            index_session.execute(
                select(Node).where(Node.snapshot_id == snapshot_id, Node.kind == "file")
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

        resolver = category_resolver or get_categories_for_path

        total_files = len(files)
        logger.info(f"Processing {total_files} files...")

        # Process each file
        folder_structure = FolderV2(name="Root")
        for i, file in enumerate(files, 1):
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{total_files} files")

            # Get categories using index session for node lookups and work session for groups
            categories = resolver(
                index_session,
                work_session,
                file.abs_path,
                iteration_id,
            )
            names = [cast(str, cat.processed_name) for cat in categories]
            new_path = "/".join(names)

            # Update or create FileMapping in work database
            existing_mapping = work_session.execute(
                select(FileMapping).where(
                    FileMapping.run_id == run_id, FileMapping.node_id == file.node_id
                )
            ).scalar_one_or_none()

            if existing_mapping:
                existing_mapping.new_path = new_path
            else:
                work_session.add(
                    FileMapping(
                        run_id=run_id,
                        node_id=file.node_id,
                        original_path=file.abs_path,
                        new_path=new_path,
                    )
                )

            category_names = [
                (category.processed_name, category.confidence)
                for category in categories
            ]
            insert_file(folder_structure, file, category_names, new_path)

        work_session.add(
            FolderStructure(
                run_id=run_id,
                structure_type=structure_type.value,
                structure=folder_structure.model_dump_json(),
            )
        )
        work_session.commit()
