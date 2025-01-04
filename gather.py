import os
import sqlite3
import json
import zipfile
from pathlib import Path
from utils.config import KNOWN_VARIANT_TOKENS
from utils.filename_utils import clean_filename, split_view_type


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
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        DROP TABLE IF EXISTS categories
                """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            renamed_category TEXT,
            folder_id INTEGER
        )
    """)

    if update_table:
        commands = [
            "ALTER table folders ADD cleaned_name TEXT",
            "ALTER table folders ADD categories TEXT",
            "ALTER table folders ADD subject TEXT",
            "ALTER table folders ADD variants TEXT",
            "ALTER table folders RENAME classification to file_source",
            "ALTER table folders ADD classification TEXT",
        ]
        for command in commands:
            try:
                cur.execute(command)
            except Exception:
                print(f'unable to execute "{command}"')

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
            cur.execute(
                """
                UPDATE folders
                SET classification = ?
                WHERE id = ?
            """,
                ("zip_file", id),
            )
            categories, variants = parse_name(row[:-4])
        else:
            categories, variants = parse_name(row)

        variants = list(set(variants))
        cleaned_name = categories[0] if categories else variants[0] if variants else ""
        cur.execute(
            """
            UPDATE folders
            SET cleaned_name = ?, categories = ?, variants = ? 
            WHERE id = ?
        """,
            (cleaned_name, ",".join(categories), ",".join(variants), id),
        )

        if categories:
            for category in categories:
                cur.execute(
                    """
                    INSERT INTO categories (category, folder_id)
                    VALUES (?, ?)
                    """,
                    (category, id),
                )

    conn.commit()
    conn.close()


def process_zip(
    zip_source,
    parent_path: str,
    zip_name: str,
    base_depth: int,
    cur: sqlite3.Cursor,
    preserve_modules: bool = True,
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
                    (zip_name, parent_path, base_depth),
                )
                return

            basename = os.path.splitext(zip_name)[0]
            # store the zipfile as a folder if it isn't redundant
            cur.execute(
                """
                INSERT INTO folders (folder_name, folder_path, parent_path, depth, classification)
                VALUES (?, ?, ?, ?, 'zip_file')
            """,
                (
                    zip_name,
                    os.path.join(parent_path, zip_name),
                    parent_path,
                    base_depth,
                ),
            )

            for entry in entries:
                parts = entry.split("/")
                if any(should_ignore(part) for part in parts):
                    continue

                # Process directory structure
                current_path = os.path.join(parent_path, zip_name)
                for part in parts[:-1]:
                    if not part:
                        continue

                    new_path = os.path.join(current_path, part)
                    if new_path not in processed_dirs and part != basename:
                        processed_dirs.add(new_path)
                        depth = new_path.count(os.sep) - base_depth

                        cur.execute(
                            """
                            INSERT INTO folders (folder_name, folder_path, parent_path, depth, classification)
                            VALUES (?, ?, ?, ?, 'zip_content')
                        """,
                            (part, new_path, current_path, depth),
                        )

                    current_path = new_path

                # Process file
                if not entry.endswith("/"):
                    file_name = parts[-1]
                    if file_name:
                        file_path = os.path.join(current_path, file_name)
                        depth = file_path.count(os.sep) - base_depth

                        # Get siblings (files in same directory)
                        dir_path = "/".join(parts[:-1]) + "/"
                        siblings = [
                            os.path.basename(e)
                            for e in entries
                            if os.path.dirname(e) + "/" == dir_path
                            and os.path.basename(e) != file_name
                        ]

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
                            INSERT INTO files (file_name, file_path, depth, siblings)
                            VALUES (?, ?, ?, ?)
                        """,
                            (file_name, file_path, depth, json.dumps(siblings)),
                        )

    except NotImplementedError | RuntimeError as e:
        print(f"Error processing zip {zip_name}: {e}")


def gather_folder_structure_and_store(base_path: str, db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if not should_ignore(d)]
        files = [f for f in files if not should_ignore(f)]

        for d in dirs:
            folder_path = os.path.join(root, d)
            depth = folder_path.count(os.sep) - base_path.count(os.sep)
            parent_path = root if root != base_path else ""

            cur.execute(
                """
                INSERT INTO folders (folder_name, folder_path, parent_path, depth, classification)
                VALUES (?, ?, ?, ?, 'filesystem')
            """,
                (d, folder_path, parent_path, depth),
            )

        for f in files:
            file_path = os.path.join(root, f)
            depth = file_path.count(os.sep) - base_path.count(os.sep)
            sibling_list = [s for s in files if s != f]

            if f.lower().endswith(".zip"):
                try:
                    process_zip(file_path, root, f, depth, cur)
                except zipfile.BadZipFile as e:
                    print(f"Error processing zip file {file_path}: {e}")
                continue

            cur.execute(
                """
                INSERT INTO files (file_name, file_path, depth, siblings)
                VALUES (?, ?, ?, ?)
            """,
                (f, file_path, depth, json.dumps(sibling_list)),
            )

    conn.commit()
    conn.close()

    clean_file_name_post(db_path=db_path, update_table=True)
