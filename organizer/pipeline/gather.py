import zipfile
import logging
from pathlib import Path
from typing import Set, List, Tuple
from sqlalchemy import select
from sqlalchemy.orm import Session
from api.api import FolderV2, StructureType
from data_models.database import (
    FolderStructure,
    setup_folder_categories,
    get_session,
    Folder as dbFolder,
    File as dbFile,
)
from utils.config import get_config
from utils.filename_utils import clean_filename
import os
from utils.folder_structure import insert_file_in_structure

logger = logging.getLogger(__name__)


def should_ignore(name: str) -> bool:
    """Check if a file or directory should be ignored."""
    config = get_config()
    return name in config.should_ignore or name.startswith("._")


def count_zip_children(entries: List[str], target_dir: str = "") -> Tuple[int, int]:
    """
    Count direct children in a zip file at a specific directory level
    Returns (folder_children, file_children)
    """
    folders = 0
    files = 0

    # Normalize target directory path
    target_dir = target_dir.rstrip("/")
    if target_dir:
        target_dir = target_dir + "/"

    # Get direct children of the target directory
    for entry in entries:
        # Skip entries not in target directory
        if not entry.startswith(target_dir):
            continue

        # Remove target directory prefix
        rel_path = entry[len(target_dir) :]
        if not rel_path:
            continue

        # Count only direct children
        if "/" not in rel_path:  # File
            files += 1
        elif rel_path.count("/") == 1 and rel_path.endswith("/"):  # Directory
            folders += 1

    return folders, files


def process_zip(
    zip_source,
    parent_path: Path,
    zip_name: str,
    base_depth: int,
    session: Session,
    preserve_modules: bool = True,
) -> None:
    try:
        with zipfile.ZipFile(zip_source, "r") as zf:
            entries = zf.namelist()
            processed_dirs: Set[Path] = set()

            # Shortcut if it's a Foundry module: i.e., if module.json exists at the root level
            matching_foundry_module = [
                entry
                for entry in entries
                if entry.lower().endswith("module.json") and entry.count("/") <= 2
            ]
            if preserve_modules and matching_foundry_module:
                # HACK: this creates an artificial folder to tag foundry modules
                folder_name = "Foundry Module " + zip_name
                new_folder = dbFolder(
                    folder_name=folder_name,
                    folder_path=str(parent_path / folder_name),
                    parent_path=str(parent_path),
                    depth=base_depth,
                    file_source="zip_file",
                    num_folder_children=0,
                    num_file_children=1,
                )
                new_file = dbFile(
                    file_name=zip_name,
                    file_path=str(parent_path / folder_name / zip_name),
                    depth=base_depth,
                )
                session.add(new_folder)
                session.add(new_file)
                session.commit()
                return

            # Count children at the ZIP root level
            root_folders, root_files = count_zip_children(entries)

            basename = Path(zip_name).stem
            matching_root_folder = [
                entry for entry in entries if entry.startswith(basename + "/")
            ]
            # Store the ZIP file as a folder with children counts if not redundant
            if not matching_root_folder:
                new_folder = dbFolder(
                    folder_name=zip_name,
                    folder_path=str(parent_path / zip_name),
                    parent_path=str(parent_path),
                    depth=base_depth,
                    file_source="zip_file",
                    num_folder_children=root_folders,
                    num_file_children=root_files,
                )
                session.add(new_folder)
                session.commit()

            for entry in entries:
                entry_path = Path(entry)
                if any(should_ignore(part) for part in entry_path.parts):
                    continue

                # Process directory structure
                current_path = parent_path / zip_name
                for part in entry_path.parts[:-1]:  # Exclude the file name itself
                    new_path = current_path / part
                    if new_path not in processed_dirs:
                        processed_dirs.add(new_path)
                        depth = len(new_path.parts) - base_depth

                        # Count children at this directory level
                        folder_children, file_children = count_zip_children(
                            entries, str(new_path.relative_to(parent_path / zip_name))
                        )

                        new_folder = dbFolder(
                            folder_name=part,
                            folder_path=str(new_path),
                            parent_path=str(current_path),
                            depth=depth,
                            file_source="zip_content",
                            num_folder_children=folder_children,
                            num_file_children=file_children,
                        )
                        session.add(new_folder)

                    current_path = new_path

                # Process file
                if not entry.endswith("/"):
                    file_name = entry_path.name
                    if file_name:
                        file_path = current_path / file_name
                        depth = len(file_path.parts) - base_depth

                        if file_name.lower().endswith(".zip"):
                            try:
                                with zf.open(entry) as nested_content:
                                    process_zip(
                                        nested_content,
                                        current_path,
                                        file_name,
                                        depth,
                                        session,
                                    )
                            except (zipfile.BadZipFile, Exception) as e:
                                logger.error(
                                    f"Error processing nested zip {entry}: {e}"
                                )
                        else:
                            new_file = dbFile(
                                file_name=file_name,
                                file_path=str(file_path),
                                depth=depth,
                            )
                            session.add(new_file)

            session.commit()

    except (NotImplementedError, zipfile.BadZipFile) as e:
        logger.error(f"Error processing zip {zip_name}: {e}")
        raise


def calculate_structure(session: Session, root_dir: Path):
    files = session.execute(select(dbFile)).scalars().all()

    total_files = len(files)
    logger.info(f"Processing {total_files} files...")

    folder_structure = FolderV2(name=str(root_dir))
    for i, file in enumerate(files, 1):
        if i % 100 == 0:
            logger.info(f"Processed {i}/{total_files} folders")

        file_path = Path(file.file_path)  # type: ignore[arg-type]  # ty bug: SQLAlchemy ORM attribute should be str
        file_path = file_path.relative_to(root_dir)
        insert_file_in_structure(folder_structure, file, file_path.parent.parts)

    session.add(
        FolderStructure(
            structure_type=StructureType.original,
            structure=folder_structure.model_dump_json(),
        )
    )


def gather_folder_structure_and_store(base_path: Path, db_path: Path) -> None:
    """Gather folder structure and store in database using SQLAlchemy."""
    session = get_session(db_path)

    with session:
        for root, dirs, files in os.walk(base_path):
            # Filter out ignored files and directories
            dirs[:] = [d for d in dirs if not should_ignore(d)]
            files = [f for f in files if not should_ignore(f)]

            for d in dirs:
                folder_path = Path(root) / d
                depth = len(folder_path.parts) - len(base_path.parts)
                parent_path = Path(root) if Path(root) != base_path else Path("")

                # Count children in this directory
                child_dirs = [
                    child_dir
                    for child_dir in os.listdir(folder_path)
                    if (folder_path / child_dir).is_dir()
                    and not should_ignore(child_dir)
                ]
                child_files = [
                    child_file
                    for child_file in os.listdir(folder_path)
                    if (folder_path / child_file).is_file()
                    and not should_ignore(child_file)
                ]

                new_folder = dbFolder(
                    folder_name=d,
                    folder_path=str(folder_path),
                    parent_path=str(parent_path),
                    depth=depth,
                    file_source="filesystem",
                    num_folder_children=len(child_dirs),
                    num_file_children=len(child_files),
                )
                session.add(new_folder)

            for f in files:
                file_path = Path(root) / f
                depth = len(file_path.parts) - len(base_path.parts)

                if f.lower().endswith(".zip"):
                    try:
                        process_zip(file_path, Path(root), f, depth, session)
                    except zipfile.BadZipFile as e:
                        logger.error(f"Error processing zip file {file_path}: {e}")
                    continue

                new_file = dbFile(
                        file_name=file_path.name,
                        file_path=str(file_path),
                        depth=depth,
                    )
                session.add(new_file)

            session.commit()

        calculate_structure(session, base_path)
        session.commit()

        # update all the folders with the cleaned filename
        folders = session.query(dbFolder).all()

        for folder in folders:
            if folder.folder_name.endswith(".zip"):
                cleaned_name = clean_filename(folder.folder_name[:-4])
            else:
                cleaned_name = clean_filename(folder.folder_name)

            # Update the cleaned name
            folder.cleaned_name = cleaned_name

        session.commit()
