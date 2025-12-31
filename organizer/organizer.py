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

file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Also redirect stderr to logging
class StderrToLogger:
    def __init__(self, logger, level=logging.ERROR):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())
        # Also write to original stderr
        sys.__stderr__.write(buf)

    def flush(self):
        pass

sys.stderr = StderrToLogger(logging.getLogger('stderr'))

logger = logging.getLogger(__name__)

app = typer.Typer()

app.add_typer(fine_tuning_app, name="model")


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
    try:
        result = export_snapshot_structure(
            output_path=output_path,
            storage_path=storage_path,
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
