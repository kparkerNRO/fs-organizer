import typer
import logging
import sys
from datetime import datetime
from pathlib import Path
from api.api import StructureType
from stages.folder_reconstruction import (
    get_folder_heirarchy,
    recalculate_cleaned_paths_for_structure,
)
from stages.gather import ingest_filesystem
from stages.grouping.group import group_folders
from stages.categorize import calculate_folder_structure
from storage.id_defaults import get_latest_run_for_snapshot
from storage.manager import StorageManager
from storage.work_models import Run
from utils.export_structure import export_snapshot_structure

from fine_tuning.cli import app as fine_tuning_app

# Configure root logger to output to both stdout and log file
log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"organizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Create handlers with explicit configuration
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

app = typer.Typer()

app.add_typer(fine_tuning_app, name="model")


def _get_latest_run(storage_manager: StorageManager, snapshot_id: int) -> Run:
    run = get_latest_run_for_snapshot(storage_manager, snapshot_id)
    if run is None:
        typer.echo("Error: no run found for snapshot.", err=True)
        raise typer.Exit(1)
    return run


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
    storage_manager = StorageManager(storage_path)
    base_path = base_path.resolve()

    snapshot_id = ingest_filesystem(storage_manager, base_path)
    typer.echo(f"✓ Created snapshot ID: {snapshot_id}")
    typer.echo("Gather complete.")


@app.command()
def group(
    storage_path: Path = typer.Option(
        None,
        "--storage",
        "-s",
        help="Storage directory (contains index.db). If not specified, uses default data directory.",
    ),
    snapshot_id: int | None = typer.Option(
        None,
        "--snapshot-id",
        help="Snapshot ID to use when selecting a run.",
    ),
    run_id: int | None = typer.Option(
        None,
        "--run-id",
        help="Run ID to use instead of looking up the latest run.",
    ),
):
    """
    Classify folders in the given run_data.db
    using known variant detection + structural heuristics.
    """
    typer.echo(f"Grouping folders in: {storage_path}")
    storage_manager = StorageManager(storage_path)
    if run_id is not None:
        if snapshot_id is None:
            with storage_manager.get_work_session() as session:
                run = session.query(Run).filter(Run.id == run_id).first()
                if run is None:
                    typer.echo("Error: run not found.", err=True)
                    raise typer.Exit(1)
                snapshot_id = run.snapshot_id
        group_folders(storage_manager, run_id=run_id, snapshot_id=snapshot_id)
        typer.echo("Grouping complete.")
        return

    if snapshot_id is None:
        typer.echo(
            "Error: snapshot_id is required to select the run for grouping.",
            err=True,
        )
        raise typer.Exit(1)

    run = _get_latest_run(storage_manager, snapshot_id)
    group_folders(storage_manager, run_id=run.id, snapshot_id=snapshot_id)
    # calculate_folder_structure(db_path)
    typer.echo("Grouping complete.")


@app.command()
def folders(
    storage_path: Path = typer.Argument(
        ..., help="Storage directory containing index.db and work.db"
    ),
    snapshot_id: int = typer.Option(..., help="Snapshot ID to process"),
    run_id: int | None = typer.Option(None, help="Run ID to use"),
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
    typer.echo(f"Generating folder hierarchy from storage: {storage_path}")
    typer.echo(f"Using snapshot_id={snapshot_id}")

    storage_manager = StorageManager(storage_path)
    if run_id is None:
        run = _get_latest_run(storage_manager, snapshot_id)
        run_id = run.id
    typer.echo(f"Using run_id={run_id}")

    if structure_type != StructureType.original:
        calculate_folder_structure(
            storage_manager, snapshot_id, run_id, structure_type=structure_type
        )
    recalculate_cleaned_paths_for_structure(
        storage_manager, snapshot_id, run_id, structure_type=structure_type
    )
    get_folder_heirarchy(storage_manager, run_id, structure_type=structure_type)
    typer.echo("Folder hierarchy generation complete.")


@app.command()
def export_structure(
    output_path: Path = typer.Argument(..., help="Output JSON file path"),
    storage_path: Path = typer.Option(
        None,
        "--storage",
        "-s",
        help="Storage directory (contains index.db). If not specified, uses default data directory.",
    ),
    include_files: bool = typer.Option(
        False,
        "--include-files",
        "-f",
        help="Include files in the directory structure (default: directories only)",
    ),
):
    """
    Export the most recent snapshot's directory structure to a JSON file.
    """
    storage_manager = StorageManager(storage_path)
    try:
        result = export_snapshot_structure(
            output_path=output_path,
            storage=storage_manager,
            include_files=include_files,
        )

        typer.echo(
            f"Exporting snapshot {result['snapshot_id']} from {result['created_at']}"
        )
        typer.echo(f"Root path: {result['root_path']}")
        typer.echo(f"Found {result['total_nodes']} nodes")
        typer.echo(f"✓ Exported directory structure to {result['output_path']}")

    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


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
    run_id: int | None = typer.Option(
        None,
        "--run-id",
        help="Run ID to use instead of looking up the latest run.",
    ),
):
    """
    Run full pipeline: gather, group, and folders.
    """
    # Run gather
    storage_manager = StorageManager(storage_path)
    snapshot_id = ingest_filesystem(storage_manager, base_path)
    typer.echo(f"✓ Created snapshot ID: {snapshot_id}")

    if run_id is not None:
        with storage_manager.get_work_session() as session:
            run = session.query(Run).filter(Run.id == run_id).first()
            if run is None:
                typer.echo("Error: run not found.", err=True)
                raise typer.Exit(1)
            if run.snapshot_id != snapshot_id:
                typer.echo(
                    "Error: run snapshot does not match the newly created snapshot.",
                    err=True,
                )
                raise typer.Exit(1)
            session.expunge(run)
    else:
        run = _get_latest_run(storage_manager, snapshot_id)

    typer.echo("Grouping folders...")
    group_folders(storage_manager, run_id=run.id, snapshot_id=snapshot_id)
    typer.echo("✓ Grouping complete.")

    typer.echo("Calculating folder structure...")
    calculate_folder_structure(
        storage_manager, snapshot_id, run.id, structure_type=StructureType.organized
    )
    recalculate_cleaned_paths_for_structure(
        storage_manager, snapshot_id, run.id, structure_type=StructureType.organized
    )
    get_folder_heirarchy(
        storage_manager, run.id, structure_type=StructureType.organized
    )
    typer.echo("✓ Folder hierarchy generation complete.")


# FastAPI endpoints

if __name__ == "__main__":
    app()
