import json
import logging
from pathlib import Path
import os
from sqlalchemy import select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
import typer
from api.api import StructureType
from data_models.database import FolderStructure
from data_models.database import (
    Folder as dbFolder,
    GroupCategoryEntry,
    get_sessionmaker,
)

logger = logging.getLogger(__name__)

Base = declarative_base()


def generate_api_folder_structure_folder(
    session: Session, file: dbFolder, working_representation={}
):
    pass


def generate_api_folder_structure_file(
    session: Session, file: str, working_representation={}
):
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
    session: Session, path: str, working_representation={}
) -> dict:
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


def main(
    db_path: str = typer.Argument(
        "organizer/outputs/latest/latest.db", help="Path to the SQLite database file"
    ),
):
    """
    Update folder paths in the database to use cleaned names.

    Args:
        db_path: Path to the SQLite database file
    """
    logger.debug(f"Current working directory: {os.getcwd()}")
    if not os.path.exists(db_path):
        raise typer.BadParameter(f"Database file not found: {db_path}")

    logger.info(f"Processing database at: {db_path}")
    get_folder_heirarchy(db_path, StructureType.organized)


if __name__ == "__main__":
    typer.run(main)
