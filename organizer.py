import os
import json
import sqlite3
import typer
from dataclasses import dataclass, field
from typing import Dict, List
from collections import Counter
from datetime import datetime
from pathlib import Path

from gather import gather_folder_structure_and_store, clean_file_name_post
from classify import classify_folders
from group import  process_groups, calculate_and_process_groups,process_pre_calculated_groups

app = typer.Typer()

# ---- 1. Setup / Create Tables ----

def setup_database(db_path: Path):
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
        classification TEXT
    )
    """)

    # Create files table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        depth INTEGER,
        siblings TEXT
    )
    """)

    # Create counts table (folder name frequency)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS counts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        folder_name TEXT NOT NULL,
        count INTEGER NOT NULL
    )
    """)

    conn.commit()
    return conn


# ---- 4. CLI Commands ----

@app.command()
def gather(
    base_path: str = typer.Argument(...),
    output_dir: str = typer.Argument(...)
):
    """
    1) Create a timestamped subfolder in output_dir,
    2) Create a run_data.db,
    3) Gather folder/file data,
    4) Insert freq counts.
    """
    base_output = Path(output_dir)
    base_output.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = base_output / timestamp_str
    run_dir.mkdir()

    db_path = run_dir / "run_data.db"
    setup_database(db_path)
    typer.echo(f"Gathering from: {base_path}")
    typer.echo(f"DB path: {db_path}")

    gather_folder_structure_and_store(base_path, db_path)
    typer.echo("Gather complete.")


@app.command()
def postprocess(db_path: str = typer.Argument(...)):
    """
    Postprocess the gather data to clean and add additional metadata
    """
    typer.echo(f"Post-processing folders in: {db_path}")
    clean_file_name_post(Path(db_path))
    typer.echo("Processing complete.")

@app.command()
def classify(
    db_path: str = typer.Argument(...)
):
    """
    Classify folders in the given run_data.db
    using known variant detection + structural heuristics.
    """
    typer.echo(f"Classifying folders in: {db_path}")
    classify_folders(Path(db_path))
    typer.echo("Classification complete.")


@app.command()
def group(
    db_path: str = typer.Argument(...)
):
    """
    Classify folders in the given run_data.db
    using known variant detection + structural heuristics.
    """
    typer.echo(f"Grouping folders in: {db_path}")

    calculate_and_process_groups(Path(db_path), threshold=90)
    # process_pre_calculated_groups(Path(db_path))
    typer.echo("Grouping complete.")

if __name__ == "__main__":
    app()
