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
        # If the parent folder is not found, try to find it in the zip content
        parent = session.query(Folder).filter(Folder.folder_path == str(parent_path)).first()

    return parent


def get_categories_for_path(
    session: Session,
    path: str | Path,
    iteration_id: int,
    # categories: list = [],
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
        for file in files:
            # get parent folder
            if file.file_name[-4:] == ".zip":
                continue

            categories = get_categories_for_path(
                session,
                file.file_path,
                iteration_id,
            )
            session.add(
                FileProcess(
                    file_id=file.id,
                    name = file.file_name,
                    groups=categories,
                    original_path=file.file_path,
                )
            )
        session.commit()


def generate_file_heirarchy_from_file_groups(
    file: FileProcess, working_representation={}
):
    categories = file.groups
    if not categories:
        return working_representation

    current_representation = working_representation
    for category in categories:
        if category not in current_representation:
            current_representation[category] = {}
        current_representation = current_representation[category]
    if "__count__" not in current_representation:
        current_representation["__count__"] = 0
    current_representation["__count__"] += 1
 
    return working_representation


def generate_folder_heirarchy(db_path: str):
    print(os.getcwd())
    if not os.path.exists(db_path):
        raise typer.BadParameter(f"Database file not found: {db_path}")

    print(f"Processing database at: {db_path}")
    sessionmaker = get_sessionmaker(db_path)
    with sessionmaker() as session:
        files = session.execute(select(FileProcess)).scalars().all()
        folder_hierarchy = {}

        for file in files:
            folder_hierarchy = generate_file_heirarchy_from_file_groups(
                file, working_representation=folder_hierarchy
            )

        json_output = json.dumps(folder_hierarchy, indent=4, sort_keys=True)
        print(json_output)
