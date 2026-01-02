"""
Responsible for post-grouping processing

Apply categories at the file level,
combine groups that were over-zealously split
generate folder paths based on that

"""

import logging
from pathlib import Path
from typing import Optional, cast

from sqlalchemy.orm import Session
from sqlalchemy import func

from api.api import FolderV2, StructureType
from data_models.database import (
    Folder,
    get_sessionmaker,
    File as dbFile,
    FolderStructure,
    GroupCategoryEntry
)

from sqlalchemy import select

from utils.folder_structure import insert_file_in_structure

logger = logging.getLogger(__name__)


def get_parent_folder(
    session: Session, parent_path: Path, zip_content=False
) -> Optional[Folder]:
    """Find the parent folder entry based on its path."""

    parent_path_str = str(parent_path)
    parent = session.query(Folder).filter(Folder.folder_path == parent_path_str).first()

    if not parent and zip_content:
        parent_path = parent_path.parent
        parent = (
            session.query(Folder).filter(Folder.folder_path == str(parent_path)).first()
        )

    return parent


def get_categories_for_path(
    session: Session,
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
    parent = get_parent_folder(session, parent_path, zip_content)

    if not parent:
        return []

    groups = (
        session.execute(
            select(GroupCategoryEntry)
            .join(Folder, GroupCategoryEntry.folder_id == Folder.id, isouter=True)
            .where(GroupCategoryEntry.iteration_id == iteration_id)
            .where(Folder.id == parent.id)
        )
        .scalars()
        .all()
    )

    categories = get_categories_for_path(session, parent_path, iteration_id)
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
    db_path: Path, structure_type: StructureType = StructureType.organized
):
    sessionmaker = get_sessionmaker(db_path)
    with sessionmaker() as session:
        files = session.execute(select(dbFile)).scalars().all()
        iteration_id = session.execute(
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
                session,
                file.file_path,
                iteration_id,
            )
            names = [cast(str, cat.processed_name) for cat in categories]
            new_path = "/".join(names)
            file.new_path = new_path
            file.groups = names

            category_names = [
                (category.processed_name, category.confidence)
                for category in categories
            ]
            insert_file_in_structure(folder_structure, file, category_names, new_path)

        session.add(
            FolderStructure(
                structure_type=structure_type,
                structure=folder_structure.model_dump_json(),
            )
        )
        session.commit()
