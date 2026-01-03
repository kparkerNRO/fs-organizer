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
from stages.categorize import get_categories_for_path
from storage.work_models import FolderStructure, GroupCategoryEntry
from storage.index_models import Node
from storage.manager import StorageManager
from utils.filename_processing import clean_filename

logger = logging.getLogger(__name__)

Base = declarative_base()


def generate_api_folder_structure_folder(
    session: Session, file: Node, working_representation: Optional[Dict] = None
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
                .join(Node, GroupCategoryEntry.folder_id == Node.node_id)
                .where(Node.abs_path == str(parent))
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

    storage = StorageManager(Path(db_path).parent)
    with storage.get_work_session() as session:
        newest_entry = session.execute(
            select(FolderStructure)
            .where(FolderStructure.structure_type == type.value)
            .order_by(FolderStructure.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        if newest_entry:
            entry = newest_entry.structure
            entry = json.dumps(json.loads(entry), indent=4)
            logger.info(entry)
            return newest_entry.structure
        return None


def _resolve_cleaned_name(folder: Node) -> str:
    # Try to get normalized name from features if available
    if folder.features and folder.features.normalized_name:
        return str(folder.features.normalized_name)
    base_name = str(folder.name)

    return clean_filename(base_name)


def _build_cleaned_path(
    folder: Node,
    folder_by_id: Dict[int, Node],
    cache: Dict[int, str],
) -> str:
    folder_id = int(folder.node_id)
    if folder_id in cache:
        return cache[folder_id]

    name = _resolve_cleaned_name(folder)
    parent_node_id = folder.parent_node_id
    if not parent_node_id:
        cleaned_path = name
    else:
        parent = folder_by_id.get(parent_node_id)
        if parent is None:
            cleaned_path = name
        else:
            parent_cleaned = _build_cleaned_path(parent, folder_by_id, cache)
            if parent_cleaned:
                cleaned_path = str(Path(parent_cleaned) / name)
            else:
                cleaned_path = name

    cache[folder_id] = cleaned_path
    return cleaned_path


def recalculate_cleaned_paths(db_path: str) -> int:
    """
    Calculate cleaned paths for folders in the new storage system.
    Note: The new storage system (Node) doesn't have a cleaned_path field,
    so this function now just returns the count of folders processed.
    Cleaned paths should be computed on-the-fly when needed.
    """
    storage = StorageManager(Path(db_path).parent)
    with storage.get_index_session(read_only=True) as session:
        folders = session.execute(
            select(Node).where(Node.kind == 'dir')
        ).scalars().all()
        folder_by_id = {folder.node_id: folder for folder in folders}
        cache: Dict[int, str] = {}

        # Build cleaned paths (but can't store them in Node)
        for folder in folders:
            _ = _build_cleaned_path(folder, folder_by_id, cache)

    return len(folders)


def recalculate_cleaned_paths_for_structure(
    db_path: str, structure_type: StructureType
) -> int:
    """
    Calculate cleaned paths for folders based on structure type.
    Note: The new storage system (Node) doesn't have a cleaned_path field,
    so this function now just returns the count of folders processed.
    """
    if structure_type == StructureType.original:
        return recalculate_cleaned_paths(db_path)

    storage = StorageManager(Path(db_path).parent)
    with storage.get_index_session(read_only=True) as index_session:
        with storage.get_work_session() as work_session:
            iteration_id = work_session.execute(
                select(func.max(GroupCategoryEntry.iteration_id))
            ).scalar_one()
            folders = index_session.execute(
                select(Node).where(Node.kind == 'dir')
            ).scalars().all()

            for folder in folders:
                categories = get_categories_for_path(
                    index_session,
                    work_session,
                    Path(str(folder.abs_path)) / "__folder__",
                    iteration_id,
                )
                # Build cleaned path but can't store it in Node
                _ = "/".join(
                    [str(category.processed_name) for category in categories]
                )

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
