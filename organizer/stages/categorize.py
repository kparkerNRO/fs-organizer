"""
Responsible for post-grouping processing

Apply categories at the file level,
combine groups that were over-zealously split
generate folder paths based on that

"""

import logging
from pathlib import Path
from typing import Optional, cast
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from sqlalchemy import func, select

from api.api import FolderV2, StructureType
from storage.index_models import Node
from storage.work_models import GroupCategoryEntry, FolderStructure
from storage.manager import StorageManager

from utils.folder_structure import insert_file_in_structure

logger = logging.getLogger(__name__)


def get_parent_folder(
    session: Session, parent_path: Path, zip_content=False
) -> Optional[Node]:
    """Find the parent folder entry based on its path."""

    parent_path_str = str(parent_path)
    parent = session.query(Node).filter(
        Node.abs_path == parent_path_str,
        Node.kind == 'dir'
    ).first()

    if not parent and zip_content:
        parent_path = parent_path.parent
        parent = session.query(Node).filter(
            Node.abs_path == str(parent_path),
            Node.kind == 'dir'
        ).first()

    return parent


def get_categories_for_path(
    index_session: Session,
    work_session: Session,
    path: str | Path,
    iteration_id: int,
) -> list[GroupCategoryEntry]:
    """
    recursively get categories for the provided path
    """
    if isinstance(path, str):
        path = Path(path)

    parent_path = path.parent
    zip_content = parent_path.match("*.zip")
    parent = get_parent_folder(index_session, parent_path, zip_content)

    if not parent:
        return []

    groups = (
        work_session.execute(
            select(GroupCategoryEntry)
            .where(GroupCategoryEntry.iteration_id == iteration_id)
            .where(GroupCategoryEntry.folder_id == parent.node_id)
        )
        .scalars()
        .all()
    )

    categories = get_categories_for_path(index_session, work_session, parent_path, iteration_id)
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
    structure_type: StructureType = StructureType.organized
):
    with manager.get_index_session(read_only=True) as index_session:
        files = index_session.execute(
            select(Node)
            .where(Node.snapshot_id == snapshot_id)
            .where(Node.kind == 'file')
        ).scalars().all()

        with manager.get_work_session() as work_session:
            iteration_id = work_session.execute(
                select(func.max(GroupCategoryEntry.iteration_id))
            ).scalar_one()
            total_files = len(files)
            logger.info(f"Processing {total_files} files...")

            # Process each file
            folder_structure = FolderV2(name="Root")
            for i, file in enumerate(files, 1):
                if i % 1000 == 0:
                    logger.info(f"Processed {i}/{total_files} files")

                categories = get_categories_for_path(
                    index_session,
                    work_session,
                    file.abs_path,
                    iteration_id,
                )
                names = [cast(str, cat.processed_name) for cat in categories]
                new_path = "/".join(names)
                # Note: Node doesn't have new_path/groups attributes - these would be stored separately

                category_names = [
                    (category.processed_name, category.confidence)
                    for category in categories
                ]

                # Create a compatibility object for insert_file_in_structure
                # This function expects attributes: id, file_name, file_path
                class FileCompat:
                    def __init__(self, node: Node):
                        self.id = node.node_id
                        self.file_name = node.name
                        self.file_path = node.abs_path

                file_compat = FileCompat(file)
                insert_file_in_structure(folder_structure, file_compat, category_names, new_path)

            work_session.add(
                FolderStructure(
                    run_id=run_id,
                    structure_type=structure_type.value,
                    structure=folder_structure.model_dump_json(),
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            )
            work_session.commit()
