import typer
import logging
import sys
from datetime import datetime
from pathlib import Path
import shutil
from api.api import StructureType
from data_models.database import setup_gather
from pipeline.folder_reconstruction import get_folder_heirarchy
from pipeline.gather import gather_folder_structure_and_store, clean_file_name_post
from pipeline.classify import classify_folders
from grouping.group import group_folders
from pipeline.categorize import calculate_folder_structure

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
    output_dir: Path = typer.Argument(
        ..., file_okay=False, dir_okay=True, writable=True, resolve_path=True
    ),
):
    """
    1) Create a timestamped subfolder in output_dir,
    2) Create a run_data.db,
    3) Gather folder/file data,
    4) Insert freq counts.
    """
    base_output = output_dir
    base_output.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = base_output / timestamp_str
    run_dir.mkdir()

    db_path = run_dir / "run_data.db"
    setup_gather(db_path)
    typer.echo(f"Gathering from: {base_path}")
    typer.echo(f"DB path: {db_path}")

    gather_folder_structure_and_store(base_path, db_path)
    clean_file_name_post(Path(db_path))

    # Set up latest directory and file
    latest_dir = base_output / "latest"
    latest_db = latest_dir / "latest.db"

    # Remove existing latest directory if it exists
    if latest_dir.exists():
        shutil.rmtree(latest_dir)

    # Create new latest directory
    latest_dir.mkdir()

    # Copy the current run's database to latest.db
    shutil.copy2(db_path, latest_db)
    typer.echo(f"Copied latest run to: {latest_db}")
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
def folders(db_path: str = typer.Argument(...)):
    """
    Generate a folder hierarchy from the cleaned paths in the database.
    """
    typer.echo(f"Generating folder hierarchy from: {db_path}")
    calculate_folder_structure(db_path)
    get_folder_heirarchy(db_path, type=StructureType.organized)
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
    output_dir: Path = typer.Argument(
        ..., file_okay=False, dir_okay=True, writable=True, resolve_path=True
    ),
):
    # Run gather
    gather(base_path, output_dir)

    # Get the path to the latest db
    latest_db = output_dir / "latest" / "latest.db"

    # Run group
    group(str(latest_db))

    # Run folders
    folders(str(latest_db))


# FastAPI endpoints

if __name__ == "__main__":
    app()
