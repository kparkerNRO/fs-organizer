"""
Responsible for post-grouping processing

Apply categories at the file level,
combine groups that were over-zealously split
generate folder paths based on that

"""

import json
import os
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func
import typer

from data_models.database import (
    Folder,
    setup_file_processing,
    get_sessionmaker,
    FileProcess,
    File,
)

from sqlalchemy import select

from grouping.group import GroupCategoryEntry


def get_parent_folder(
    session: Session, parent_path: Path, zip_content=False
) -> Optional[Folder]:
    """Find the parent folder entry based on its path."""

    parent_path_str = str(parent_path)
    parent = session.query(Folder).filter(Folder.folder_path == parent_path_str).first()

    if not parent and zip_content:
        parent_path = parent_path.parent
        parent = session.query(Folder).filter(Folder.folder_path == str(parent_path)).first()

    return parent


def get_categories_for_path(
    session: Session,
    path: str | Path,
    iteration_id: int,
):
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
            select(GroupCategoryEntry.processed_name)
            .join(Folder, GroupCategoryEntry.folder_id == Folder.id, isouter=True)
            .where(GroupCategoryEntry.iteration_id == iteration_id - 1)
            .where(Folder.id == parent.id)
        )
        .scalars()
        .all()
    )

    
    categories =  get_categories_for_path(session,parent_path, iteration_id )
    filtered_categories = [group for group in groups if group not in categories]
    merged_groups = categories + filtered_categories

    return merged_groups

def calculate_categories(db_path: Path):
    setup_file_processing(db_path)

    sessionmaker = get_sessionmaker(db_path)
    with sessionmaker() as session:
        files = session.execute(select(File)).scalars().all()
        iteration_id = session.execute(
            select(func.max(GroupCategoryEntry.iteration_id))
        ).scalar_one()
        total_files = len(files)
        print(f"Processing {total_files} files...")

        # Process each file
        for i, file in enumerate(files, 1):
            if i % 100 == 0:
                print(f"Processed {i}/{total_files} folders")

            # get parent folder
            if file.file_name[-4:] == ".zip":
                continue

            categories = get_categories_for_path(
                session,
                file.file_path,
                iteration_id,
            )
            new_path = "/".join(categories)
            session.add(
                FileProcess(
                    file_id=file.id,
                    name = file.file_name,
                    groups=categories,
                    original_path=file.file_path,
                    new_path=new_path
                )
            )
        session.commit()

