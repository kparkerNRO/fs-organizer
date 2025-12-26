import typer
import logging
import sys
from pathlib import Path
from api.api import StructureType
from stages.folder_reconstruction import (
    get_folder_heirarchy,
    recalculate_cleaned_paths_for_structure,
)
from stages.gather import ingest_filesystem
from stages.grouping.group import group_folders
from stages.categorize import calculate_folder_structure

# Configure root logger to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def gather(
    base_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    storage_path: Path = typer.Option(
        None,
        "--storage",
        "-s",
        help="Storage directory (contains index.db). If not specified, uses default data directory.",
    ),
):
    """
    Scan filesystem and create immutable snapshot in index.db.
    """
    typer.echo(f"Gathering from: {base_path}")

    snapshot_id = ingest_filesystem(base_path, storage_path)
    typer.echo(f"✓ Created snapshot ID: {snapshot_id}")
    typer.echo("Gather complete.")


@app.command()
def group(db_path: str = typer.Argument(...)):
    """
    Classify folders in the given run_data.db
    using known variant detection + structural heuristics.
    """
    typer.echo(f"Grouping folders in: {db_path}")
    group_folders(Path(db_path))
    calculate_folder_structure(db_path)
    typer.echo("Grouping complete.")


@app.command()
def folders(
    db_path: str = typer.Argument(...),
    structure_type: StructureType = typer.Option(
        StructureType.organized,
        "--structure-type",
        "-s",
        help="Folder structure type to generate and use for cleaned paths.",
    ),
):
    """
    Generate a folder hierarchy from the cleaned paths in the database.
    """
    typer.echo(f"Generating folder hierarchy from: {db_path}")
    if structure_type != StructureType.original:
        calculate_folder_structure(db_path, structure_type=structure_type)
    recalculate_cleaned_paths_for_structure(db_path, structure_type=structure_type)
    get_folder_heirarchy(db_path, type=structure_type)
    typer.echo("Folder hierarchy generation complete.")


@app.command()
def pipeline(
    base_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    storage_path: Path = typer.Option(
        None,
        "--storage",
        "-s",
        help="Storage directory (contains index.db). If not specified, uses default data directory.",
    ),
):
    """
    Run full pipeline: gather, group, and folders.

    NOTE: Currently only gather is implemented with new storage.
    Group and folders commands need to be migrated to work with snapshots.
    """
    # Run gather
    snapshot_id = ingest_filesystem(base_path, storage_path)
    typer.echo(f"✓ Created snapshot ID: {snapshot_id}")

    # TODO: Update group and folders commands to work with snapshot_id
    typer.echo(
        "⚠ Pipeline incomplete: group and folders commands not yet migrated to new storage"
    )
    typer.echo(
        f"  Run 'group' and 'folders' commands manually with snapshot_id={snapshot_id}"
    )


# FastAPI endpoints

if __name__ == "__main__":
    app()
