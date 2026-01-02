from datetime import datetime, timezone
import zipfile
import logging
from pathlib import Path
from typing import Set, List, Tuple
from sqlalchemy import select
from sqlalchemy.orm import Session
from api.api import FolderV2, StructureType
from data_models.database import (
    FolderStructure,
    get_session,
    Folder as dbFolder,
    File as dbFile,
)
from utils.config import get_config
from utils.filename_processing import clean_filename
import os
from utils.folder_structure import insert_file_in_structure
from storage.manager import NodeKind, FileSource
from storage.index_models import Node, NodeFeatures

logger = logging.getLogger(__name__)


def should_ignore(name: str) -> bool:
    """Check if a file or directory should be ignored."""
    config = get_config()
    return name in config.should_ignore or name.startswith("._")


def is_valid_zip(file_path: Path) -> tuple[bool, str | None]:
    """
    Check if a file is actually a valid ZIP archive.
    Returns (is_valid, error_message).
    """
    try:
        with open(file_path, "rb") as f:
            magic = f.read(16)

        # ZIP files start with PK
        if magic[:2] == b"PK":
            return True, None

        # Check for common fake zip file types
        if magic.startswith(b"<!DOCTYPE") or magic.startswith(b"<html"):
            # Read a bit more to identify the type
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(2000)
                    if "Dropbox" in content and "File Deleted" in content:
                        return False, "HTML file (Dropbox deleted/expired link)"
                    elif "Dropbox" in content:
                        return False, "HTML file (Dropbox error page)"
                    elif "MyAirBridge" in content or "myairbridge" in content:
                        return False, "HTML file (MyAirBridge download page)"
                    else:
                        return False, "HTML file"
            except Exception:
                return False, "HTML file"

        # Empty file
        if len(magic) == 0 or file_path.stat().st_size == 0:
            return False, "Empty file (0 bytes)"

        # Unknown format
        return False, f"Not a ZIP file (starts with {magic[:4].hex()})"

    except Exception as e:
        return False, f"Error reading file: {e}"


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
    base_path: Path,
    parent_rel_path: Path,
    parent_node_id: int | None,
    zip_name: str,
    session: Session,
    snapshot_id: int,
    zip_file_source: FileSource = FileSource.ZIP_FILE,
    preserve_modules: bool = True,
) -> None:
    try:
        with zipfile.ZipFile(zip_source, "r") as zf:
            entries = zf.namelist()
            zip_rel_path = (
                parent_rel_path / zip_name if parent_rel_path else Path(zip_name)
            )
            zip_abs_path = base_path / zip_rel_path

            # Shortcut if it's a Foundry module: i.e., if module.json exists at the root level
            matching_foundry_module = [
                entry
                for entry in entries
                if entry.lower().endswith("module.json") and entry.count("/") <= 2
            ]
            if preserve_modules and matching_foundry_module:
                # HACK: this creates an artificial folder to tag foundry modules
                folder_name = "Foundry Module " + zip_name
                folder_rel_path = (
                    parent_rel_path / folder_name
                    if parent_rel_path
                    else Path(folder_name)
                )
                folder_depth = len(folder_rel_path.parts)
                folder_node = _create_node(
                    session,
                    snapshot_id=snapshot_id,
                    kind=NodeKind.DIR,
                    name=folder_name,
                    rel_path=folder_rel_path,
                    abs_path=base_path / folder_rel_path,
                    depth=folder_depth,
                    parent_node_id=parent_node_id,
                    file_source=zip_file_source,
                    num_folder_children=0,
                    num_file_children=1,
                )
                _create_node(
                    session,
                    snapshot_id=snapshot_id,
                    kind=NodeKind.FILE,
                    name=zip_name,
                    rel_path=folder_rel_path / zip_name,
                    abs_path=base_path / folder_rel_path / zip_name,
                    depth=folder_depth + 1,
                    parent_node_id=folder_node.node_id,
                    file_source=zip_file_source,
                )
                session.commit()
                return

            # Count children at the ZIP root level
            root_folders, root_files = count_zip_children(entries)

            zip_node = _create_node(
                session,
                snapshot_id=snapshot_id,
                kind=NodeKind.FILE,
                name=zip_name,
                rel_path=zip_rel_path,
                abs_path=zip_abs_path,
                depth=len(zip_rel_path.parts),
                parent_node_id=parent_node_id,
                file_source=zip_file_source,
                num_folder_children=root_folders,
                num_file_children=root_files,
            )
            zip_dir_nodes: dict[Path, int] = {}

            for entry in entries:
                entry_path = Path(entry)
                if any(should_ignore(part) for part in entry_path.parts):
                    continue

                # Process directory structure
                current_rel_path = zip_rel_path
                current_parent_id = zip_node.node_id
                for part in entry_path.parts[:-1]:  # Exclude the file name itself
                    new_rel_path = current_rel_path / part
                    if new_rel_path in zip_dir_nodes:
                        current_parent_id = zip_dir_nodes[new_rel_path]
                    else:
                        depth = len(new_rel_path.parts)

                        # Count children at this directory level
                        folder_children, file_children = count_zip_children(
                            entries, str(new_rel_path.relative_to(zip_rel_path))
                        )

                        new_folder = _create_node(
                            session,
                            snapshot_id=snapshot_id,
                            kind=NodeKind.DIR,
                            name=part,
                            rel_path=new_rel_path,
                            abs_path=base_path / new_rel_path,
                            depth=depth,
                            parent_node_id=current_parent_id,
                            file_source=FileSource.ZIP_CONTENT,
                            num_folder_children=folder_children,
                            num_file_children=file_children,
                        )
                        current_parent_id = new_folder.node_id
                        zip_dir_nodes[new_rel_path] = new_folder.node_id

                    current_rel_path = new_rel_path

                # Process file
                if not entry.endswith("/"):
                    file_name = entry_path.name
                    if file_name:
                        file_rel_path = current_rel_path / file_name
                        depth = len(file_rel_path.parts)

                        if file_name.lower().endswith(".zip"):
                            try:
                                with zf.open(entry) as nested_content:
                                    process_zip(
                                        nested_content,
                                        base_path,
                                        current_rel_path,
                                        current_parent_id,
                                        file_name,
                                        session,
                                        snapshot_id,
                                        FileSource.ZIP_CONTENT,
                                    )
                            except (zipfile.BadZipFile, Exception) as e:
                                logger.error(
                                    f"Error processing nested zip {entry}: {e}"
                                )
                        else:
                            info = zf.getinfo(entry)
                            mtime = datetime(
                                *info.date_time, tzinfo=timezone.utc
                            ).timestamp()
                            _create_node(
                                session,
                                snapshot_id=snapshot_id,
                                kind=NodeKind.FILE,
                                name=file_name,
                                rel_path=file_rel_path,
                                abs_path=base_path / file_rel_path,
                                depth=depth,
                                parent_node_id=current_parent_id,
                                file_source=FileSource.ZIP_CONTENT,
                                size=info.file_size,
                                mtime=mtime,
                            )

            session.commit()

    except (NotImplementedError, zipfile.BadZipFile) as e:
        logger.error(f"Error processing zip {zip_name}: {e}")
        raise


def _process_zip_legacy(
    zip_source,
    parent_path: Path,
    zip_name: str,
    base_depth: int,
    session: Session,
    preserve_modules: bool = True,
) -> None:
    """Legacy ZIP processing for run_data.db folders/files tables."""
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
                                    _process_zip_legacy(
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


def _create_node(
    session: Session,
    *,
    snapshot_id: int,
    kind: NodeKind,
    name: str,
    rel_path: Path,
    abs_path: Path,
    depth: int,
    parent_node_id: int | None,
    file_source: FileSource,
    num_folder_children: int = 0,
    num_file_children: int = 0,
    size: int | None = None,
    mtime: float | None = None,
    ctime: float | None = None,
    inode: int | None = None,
) -> Node:
    node = Node(
        snapshot_id=snapshot_id,
        parent_node_id=parent_node_id,
        kind=kind.value,
        name=name,
        rel_path=str(rel_path),
        abs_path=str(abs_path),
        ext=Path(name).suffix or None,
        size=size,
        mtime=mtime,
        ctime=ctime,
        inode=inode,
        depth=depth,
        file_source=file_source.value,
        num_folder_children=num_folder_children,
        num_file_children=num_file_children,
    )
    session.add(node)
    session.flush()

    features = NodeFeatures(
        node_id=node.node_id,
        normalized_name=clean_filename(name),
    )
    session.add(features)
    return node


def ingest_filesystem(storage_manager, base_path, storage_path: Path | None):
    with storage_manager.ingestion_job(root_path=base_path) as job:
        snapshot_id = job.snapshot_id
        with storage_manager.get_index_session() as index_session:
            dir_nodes: dict[str, int] = {}

            for root, dirs, files in os.walk(base_path):
                # Filter out ignored files and directories
                dirs[:] = [d for d in dirs if not should_ignore(d)]
                files = [f for f in files if not should_ignore(f)]

                root_path = Path(root)

                for d in dirs:
                    folder_path = root_path / d
                    rel_path = folder_path.relative_to(base_path)
                    parent_rel_path = rel_path.parent
                    parent_node_id = (
                        dir_nodes.get(str(parent_rel_path))
                        if parent_rel_path != Path(".")
                        else None
                    )
                    depth = len(rel_path.parts)

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

                    stat = folder_path.stat()
                    node = _create_node(
                        index_session,
                        snapshot_id=snapshot_id,
                        kind=NodeKind.DIR,
                        name=d,
                        rel_path=rel_path,
                        abs_path=folder_path,
                        depth=depth,
                        parent_node_id=parent_node_id,
                        file_source=FileSource.FILESYSTEM,
                        num_folder_children=len(child_dirs),
                        num_file_children=len(child_files),
                        size=stat.st_size,
                        mtime=stat.st_mtime,
                        ctime=stat.st_ctime,
                        inode=stat.st_ino,
                    )
                    dir_nodes[str(rel_path)] = node.node_id

                for f in files:
                    file_path = root_path / f
                    rel_path = file_path.relative_to(base_path)
                    parent_rel_path = rel_path.parent
                    parent_node_id = (
                        dir_nodes.get(str(parent_rel_path))
                        if parent_rel_path != Path(".")
                        else None
                    )
                    depth = len(rel_path.parts)

                    if f.lower().endswith(".zip"):
                        # Validate that it's actually a ZIP file before processing
                        is_valid, error_msg = is_valid_zip(file_path)
                        if not is_valid:
                            logger.warning(f"Skipping {file_path}: {error_msg}")
                            continue

                        try:
                            process_zip(
                                file_path,
                                base_path,
                                Path("")
                                if parent_rel_path == Path(".")
                                else parent_rel_path,
                                parent_node_id,
                                f,
                                index_session,
                                snapshot_id,
                                FileSource.ZIP_FILE,
                            )
                        except zipfile.BadZipFile as e:
                            logger.error(f"Error processing zip file {file_path}: {e}")
                        continue

                    stat = file_path.stat()
                    _create_node(
                        index_session,
                        snapshot_id=snapshot_id,
                        kind=NodeKind.FILE,
                        name=f,
                        rel_path=rel_path,
                        abs_path=file_path,
                        depth=depth,
                        parent_node_id=parent_node_id,
                        file_source=FileSource.FILESYSTEM,
                        size=stat.st_size,
                        mtime=stat.st_mtime,
                        ctime=stat.st_ctime,
                        inode=stat.st_ino,
                    )

            index_session.commit()

    return snapshot_id


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
                    # Validate that it's actually a ZIP file before processing
                    is_valid, error_msg = is_valid_zip(file_path)
                    if not is_valid:
                        logger.warning(f"Skipping {file_path}: {error_msg}")
                        continue

                    try:
                        _process_zip_legacy(file_path, Path(root), f, depth, session)
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
            cleaned_name = clean_filename(folder.folder_name)
            folder.cleaned_name = cleaned_name

        session.commit()
