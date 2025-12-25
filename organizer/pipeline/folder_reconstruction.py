import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional
from sqlalchemy import func, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
import typer
from api.api import StructureType
from pipeline.categorize import get_categories_for_path
from data_models.database import FolderStructure
from data_models.database import (
    Folder as dbFolder,
    GroupCategoryEntry,
    get_sessionmaker,
)
from utils.filename_utils import clean_filename

logger = logging.getLogger(__name__)

Base = declarative_base()


def generate_api_folder_structure_folder(
    session: Session, file: dbFolder, working_representation: Optional[Dict] = None
):
    if working_representation is None:
        working_representation = {}
    pass


def generate_api_folder_structure_file(
    session: Session, file: str, working_representation: Optional[Dict] = None
):
    if working_representation is None:
        working_representation = {}
    """
    class File(BaseModel):
    id: int
    name: str
    confidence: int = 100
    possibleClassifications: list[str] = []
    originalPath: str
    newPath: str
    """
    file_path = Path(file)

    for parent in file_path.parents:
        _ = (
            session.execute(
                select(GroupCategoryEntry)
                .join(dbFolder, GroupCategoryEntry.folder_id == dbFolder.id)
                .where(dbFolder.folder_path == str(parent))
            )
            .scalars()
            .all()
        )


def generate_api_heirarchy(session: Session, column):
    files = session.execute(select(column)).scalars().all()
    folder_hierarchy = {}

    for file in files:
        folder_hierarchy = generate_folder_heirarchy_from_path(
            session, file, working_representation=folder_hierarchy
        )


def generate_folder_heirarchy_from_path(
    session: Session, path: str, working_representation: Optional[Dict] = None
) -> dict:
    if working_representation is None:
        working_representation = {}
    cleaned_path = path
    if not cleaned_path:
        return working_representation
    cleaned_parts = cleaned_path.split("/")

    # Initialize the working representation with the cleaned path
    current_representation = working_representation
    for part in cleaned_parts:
        if part not in current_representation:
            current_representation[part] = {}
        current_representation = current_representation[part]
    if "__count__" not in current_representation:
        current_representation["__count__"] = 0
    current_representation["__count__"] += 1

    return working_representation


def generate_folder_heirarchy_session(session: Session, column):
    files = session.execute(select(column)).scalars().all()
    folder_hierarchy = {}

    for file in files:
        folder_hierarchy = generate_folder_heirarchy_from_path(
            session, file, working_representation=folder_hierarchy
        )

    return folder_hierarchy


def get_folder_heirarchy(db_path: str, type: StructureType):
    logger.debug(f"Current working directory: {os.getcwd()}")
    if not os.path.exists(db_path):
        raise typer.BadParameter(f"Database file not found: {db_path}")

    sessionmaker = get_sessionmaker(db_path)
    with sessionmaker() as session:
        newest_entry = session.execute(
            select(FolderStructure)
            .where(FolderStructure.structure_type == type)
            .order_by(FolderStructure.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        entry = newest_entry.structure
        entry = json.dumps(json.loads(entry), indent=4)
        logger.info(entry)
        return newest_entry.structure


def _resolve_cleaned_name(folder: dbFolder) -> str:
    if folder.cleaned_name:
        return folder.cleaned_name
    base_name = folder.folder_name
    if base_name.endswith(".zip"):
        base_name = base_name[:-4]
    return clean_filename(base_name)


def _build_cleaned_path(
    folder: dbFolder,
    folder_by_path: Dict[str, dbFolder],
    cache: Dict[int, str],
) -> str:
    if folder.id in cache:
        return cache[folder.id]

    name = _resolve_cleaned_name(folder)
    if not folder.parent_path:
        cleaned_path = name
    else:
        parent = folder_by_path.get(folder.parent_path)
        if parent is None:
            cleaned_path = name
        else:
            parent_cleaned = _build_cleaned_path(parent, folder_by_path, cache)
            if parent_cleaned:
                cleaned_path = str(Path(parent_cleaned) / name)
            else:
                cleaned_path = name

    cache[folder.id] = cleaned_path
    return cleaned_path


def recalculate_cleaned_paths(db_path: str) -> int:
    sessionmaker = get_sessionmaker(db_path)
    with sessionmaker() as session:
        folders = session.execute(select(dbFolder)).scalars().all()
        folder_by_path = {folder.folder_path: folder for folder in folders}
        cache: Dict[int, str] = {}

        for folder in folders:
            folder.cleaned_path = _build_cleaned_path(folder, folder_by_path, cache)

        session.commit()

    return len(folders)


def recalculate_cleaned_paths_for_structure(
    db_path: str, structure_type: StructureType
) -> int:
    if structure_type == StructureType.original:
        return recalculate_cleaned_paths(db_path)

    sessionmaker = get_sessionmaker(db_path)
    with sessionmaker() as session:
        iteration_id = session.execute(
            select(func.max(GroupCategoryEntry.iteration_id))
        ).scalar_one()
        folders = session.execute(select(dbFolder)).scalars().all()

        for folder in folders:
            categories = get_categories_for_path(
                session,
                Path(folder.folder_path) / "__folder__",
                iteration_id,
            )
            folder.cleaned_path = "/".join(
                [category.processed_name for category in categories]
            )

        session.commit()

    return len(folders)


def main(
    db_path: str = typer.Argument(
        "outputs/latest/latest.db", help="Path to the SQLite database file"
    ),
):
    """
    Update folder cleaned paths in the database to use recalculated names.

    Args:
        db_path: Path to the SQLite database file
    """
    logger.debug(f"Current working directory: {os.getcwd()}")
    if not os.path.exists(db_path):
        raise typer.BadParameter(f"Database file not found: {db_path}")

    logger.info(f"Processing database at: {db_path}")
    get_folder_heirarchy(db_path, StructureType.organized)
    updated_count = recalculate_cleaned_paths(db_path)
    logger.info(f"Updated cleaned_path for {updated_count} folders")


if __name__ == "__main__":
    typer.run(main)
