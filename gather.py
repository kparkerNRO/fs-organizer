import zipfile
from pathlib import Path
from typing import Set, List
from sqlalchemy.orm import Session
from database import (
    setup_collections,
    get_session,
    Folder,
    File,
    Category,
)
from utils.config import KNOWN_VARIANT_TOKENS
from utils.filename_utils import clean_filename, split_view_type
import os

def should_ignore(name: str) -> bool:
    """Check if a file or directory should be ignored."""
    ignore_patterns = {
        "__MACOSX",
        ".DS_Store",
        "Thumbs.db",
        ".AppleDouble",
        ".LSOverride",
        "desktop.ini",
        ".fseventsd",
        ".Spotlight-V100",
        ".TemporaryItems",
        ".Trashes",
        ".DocumentRevisions-V100",
        "thumbs",
    }
    return name in ignore_patterns or name.startswith("._")

def clean_file_name_post(db_path: Path, update_table: bool = False) -> None:
    """Clean and update folder names in the database using SQLAlchemy."""
    setup_collections(db_path)
    session = get_session(db_path)

    try:
        # Fetch all folders
        folders = session.query(Folder).all()

        for folder in folders:
            if folder.folder_name.endswith('.zip'):
                cleaned_name = clean_filename(folder.folder_name[:-4])
            else:
                cleaned_name = clean_filename(folder.folder_name)

            # Update the cleaned name
            folder.cleaned_name = cleaned_name

        session.commit()
    finally:
        session.close()

def process_zip(
    zip_source,
    parent_path: Path,
    zip_name: str,
    base_depth: int,
    session: Session,
    preserve_modules: bool = True,
    num_siblings: int = 0,
) -> None:
    """Process a zip file and store its contents in the database using SQLAlchemy."""
    try:
        with zipfile.ZipFile(zip_source, "r") as zf:
            entries = zf.namelist()
            processed_dirs: Set[Path] = set()

            # shortcut for foundry modules
            if preserve_modules and "module.json" in entries:
                new_file = File(
                    file_name=zip_name,
                    file_path=str(parent_path),
                    depth=base_depth
                )
                session.add(new_file)
                session.commit()
                return

            basename = Path(zip_name).stem
            # store the zipfile as a folder
            new_folder = Folder(
                folder_name=zip_name,
                folder_path=str(parent_path / zip_name),
                parent_path=str(parent_path),
                depth=base_depth,
                file_source='zip_file',
                num_siblings=num_siblings
            )
            session.add(new_folder)
            session.commit()

            for entry in entries:
                entry_path = Path(entry)
                if any(should_ignore(part) for part in entry_path.parts):
                    continue

                # Process directory structure
                current_path = parent_path / zip_name
                for part in entry_path.parts[:-1]:
                    if not part:
                        continue

                    new_path = current_path / part
                    if new_path not in processed_dirs and part != basename:
                        processed_dirs.add(new_path)
                        depth = len(new_path.parts) - base_depth

                        # Get siblings
                        dir_path = entry_path.parent
                        zip_depth = len(entry_path.parts)
                        siblings = [
                            Path(e).name
                            for e in entries
                            if Path(e).parent == dir_path
                            and e != entry
                            and e.endswith("/")
                            and len(Path(e).parts) < zip_depth
                        ]

                        new_folder = Folder(
                            folder_name=part,
                            folder_path=str(new_path),
                            parent_path=str(current_path),
                            depth=depth,
                            file_source='zip_content',
                            num_siblings=len(siblings)
                        )
                        session.add(new_folder)

                    current_path = new_path

                # Process file
                if not entry_path.is_dir():
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
                                print(f"Error processing nested zip {entry}: {e}")

                        new_file = File(
                            file_name=file_name,
                            file_path=str(file_path),
                            depth=depth
                        )
                        session.add(new_file)

            session.commit()

    except (NotImplementedError, zipfile.BadZipFile) as e:
        print(f"Error processing zip {zip_name}: {e}")
        session.rollback()

def gather_folder_structure_and_store(base_path: Path, db_path: Path) -> None:
    """Gather folder structure and store in database using SQLAlchemy."""
    session = get_session(db_path)

    try:
        for root, dirs, files in os.walk(base_path):
            dirs[:] = [d for d in dirs if not should_ignore(d)]
            files = [f for f in files if not should_ignore(f)]

            for d in dirs:
                folder_path = Path(root) / d
                depth = len(folder_path.parts) - len(base_path.parts)
                parent_path = Path(root) if Path(root) != base_path else Path("")
                sibling_list = [s for s in dirs if s != d]

                new_folder = Folder(
                    folder_name=d,
                    folder_path=str(folder_path),
                    parent_path=str(parent_path),
                    depth=depth,
                    file_source='filesystem',
                    num_siblings=len(sibling_list)
                )
                session.add(new_folder)

            for f in files:
                file_path = Path(root) / f
                depth = len(file_path.parts) - len(base_path.parts)
                
                if f.lower().endswith(".zip"):
                    try:
                        process_zip(file_path, Path(root), f, depth, session)
                    except zipfile.BadZipFile as e:
                        print(f"Error processing zip file {file_path}: {e}")
                    continue

                new_file = File(
                    file_name=f,
                    file_path=str(file_path),
                    depth=depth
                )
                session.add(new_file)

            session.commit()

    except Exception as e:
        print(f"Error gathering folder structure: {e}")
        session.rollback()
        raise
    finally:
        session.close()

    clean_file_name_post(db_path=db_path, update_table=True)