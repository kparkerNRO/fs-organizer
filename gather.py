import sqlite3
import json
import zipfile
from pathlib import Path
from database import setup_collections
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


def clean_file_name_post(db_path, update_table: bool = False):
    setup_collections(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    rows = [
        (row[0], row[1])
        for row in cur.execute(
            """
        SELECT id, folder_name FROM folders
        """
        ).fetchall()
    ]

    def parse_name(name):
        # do an initial clean up
        categories = []
        variants = []

        clean_row = clean_filename(name)

        # handle any sub-categories in the name
        components = clean_row.split("-")
        cleaned_components = [clean_filename(component) for component in components]

        for component in cleaned_components:
            category_current = component
            while category_current:
                category, variant = split_view_type(
                    category_current, KNOWN_VARIANT_TOKENS
                )
                if category and category not in categories:
                    categories.append(category)
                if variant:
                    variants.append(variant)

                if category_current == category:
                    break

                category_current = category

        return categories, variants

    for id, row in rows:
        if row[-4:] == ".zip":
            cleaned_name = clean_filename(row[:-4])
        else:
            cleaned_name = clean_filename(row)

        cur.execute(
            """
            UPDATE folders
            SET cleaned_name = ?
            WHERE id = ?
        """,
            (cleaned_name, id),
        )

        # if categories:
        #     for category in categories:
        #         cur.execute(
        #             """
        #             INSERT INTO categories (category, folder_id)
        #             VALUES (?, ?)
        #             """,
        #             (category, id),
        #         )

    conn.commit()
    conn.close()


def process_zip(
    zip_source,
    parent_path: Path,
    zip_name: str,
    base_depth: int,
    cur: sqlite3.Cursor,
    preserve_modules: bool = True,
    num_siblings: int = 0,
) -> None:
    try:
        with zipfile.ZipFile(zip_source, "r") as zf:
            entries = zf.namelist()
            processed_dirs = set()

            # shortcut for foundry modules
            if preserve_modules and "module.json" in entries:
                cur.execute(
                    """
                    INSERT INTO files (file_name, file_path, depth)
                    VALUES (?, ?, ?)
                """,
                    (zip_name, str(parent_path), base_depth),
                )
                return

            basename = Path(zip_name).stem
            # store the zipfile as a folder if it isn't redundant
            cur.execute(
                """
                INSERT INTO folders (folder_name, folder_path, parent_path, depth, file_source, num_siblings)
                VALUES (?, ?, ?, ?, 'zip_file', num_siblings)
            """,
                (
                    zip_name,
                    str(parent_path / zip_name),
                    str(parent_path),
                    base_depth,
                    num_siblings
                ),
            )

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

                        # Get siblings (folders in same directory)
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

                        cur.execute(
                            """
                            INSERT INTO folders (
                                folder_name, 
                                folder_path, 
                                parent_path, 
                                depth, 
                                file_source, 
                                num_siblings)
                            VALUES (?, ?, ?, ?, 'zip_content', ?)
                        """,
                            (part, str(new_path), str(current_path), depth, len(siblings)),
                        )

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
                                        cur,
                                    )
                            except (zipfile.BadZipFile, Exception) as e:
                                print(f"Error processing nested zip {entry}: {e}")

                        cur.execute(
                            """
                            INSERT INTO files (file_name, file_path, depth)
                            VALUES (?, ?, ?)
                        """,
                            (file_name, str(file_path), depth),
                        )

    except (NotImplementedError, zipfile.BadZipFile) as e:
        print(f"Error processing zip {zip_name}: {e}")


def gather_folder_structure_and_store(base_path: Path, db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if not should_ignore(d)]
        files = [f for f in files if not should_ignore(f)]

        for d in dirs:
            folder_path = Path(root) / d
            depth = len(folder_path.parts) - len(base_path.parts)
            parent_path = Path(root) if Path(root) != base_path else Path("")
            sibling_list = [s for s in dirs if s != d]

            cur.execute(
                """
                INSERT INTO folders (folder_name, folder_path, parent_path, depth, file_source,num_siblings)
                VALUES (?, ?, ?, ?, 'filesystem',?)
            """,
                (d, str(folder_path), str(parent_path), depth, len(sibling_list)),
            )

        for f in files:
            file_path = Path(root) / f
            depth = len(file_path.parts) - len(base_path.parts)
            
            if f.lower().endswith(".zip"):
                try:
                    process_zip(file_path, Path(root), f, depth, cur)
                except zipfile.BadZipFile as e:
                    print(f"Error processing zip file {file_path}: {e}")
                continue

            cur.execute(
                """
                INSERT INTO files (file_name, file_path, depth)
                VALUES (?, ?, ?)
            """,
                (f, str(file_path), depth),
            )

    conn.commit()
    conn.close()

    clean_file_name_post(db_path=db_path, update_table=True)
