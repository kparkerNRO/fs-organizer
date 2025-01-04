from pathlib import Path
import sqlite3



def setup_gather(db_path: Path):
    """Create or open the SQLite database, ensuring tables exist."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create folders table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS folders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        folder_name TEXT NOT NULL,
        folder_path TEXT NOT NULL,
        parent_path TEXT,
        depth INTEGER,
        cleaned_name TEXT,
        categories TEXT,
        subject TEXT,
        variants TEXT,
        classification TEXT,
        file_source TEXT,
        siblings TEXT
    )
    """)

    # Create files table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        depth INTEGER
    )
    """)

    conn.commit()
    conn.close()

def setup_group(db_path: Path):
    """Create or open the SQLite database, ensuring tables exist."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create groups table
    # create the tables
    cur.execute("""
        DROP TABLE IF EXISTS cleaned_groups
                """)
    cur.execute("""
        DROP TABLE IF EXISTS processed_names
                """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS processed_names (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id INTEGER,
            folder_name TEXT,
            grouped_name TEXT,
            confidence FLOAT
        )
    """)

    cur.execute("""
        DROP TABLE IF EXISTS related_groups
                """)
    cur.execute("""
        DROP TABLE IF EXISTS groups
                """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_name TEXT,
            cannonical_name TEXT
        )
    """)

    conn.commit()
    conn.close()

def setup_collections(db_path: Path):
    """Create or open the SQLite database, ensuring tables exist."""
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

    conn.commit()
    conn.close()