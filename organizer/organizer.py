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
from fine_tuning.training_utils import (
    select_training_samples,
    write_sample_csv,
    read_classification_csv,
    validate_all_labels_present,
    validate_label_values,
)
from storage.manager import StorageManager
from storage.index_models import Snapshot
from storage.training_manager import get_or_create_training_session
from fine_tuning.feature_extraction import extract_features
from utils.config import get_config
from sqlalchemy import select as sql_select

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


# Training data generation commands
training_app = typer.Typer(help="Training data generation and management")
app.add_typer(training_app, name="training")

# Fine-tuning model commands (train, predict, etc.)
from fine_tuning.cli import app as fine_tuning_app
app.add_typer(fine_tuning_app, name="model")


@training_app.command("select-data")
def select_data(
    output_csv: Path = typer.Argument(..., help="Output CSV file path"),
    storage_path: Path = typer.Option(
        None,
        "--storage",
        "-s",
        help="Storage directory (contains index.db). Defaults to data/",
    ),
    snapshot_id: int = typer.Option(
        None, "--snapshot-id", help="Snapshot ID to sample from. Uses most recent if not specified."
    ),
    sample_size: int = typer.Option(
        200, "--sample-size", "-n", help="Target number of samples to select"
    ),
    min_depth: int = typer.Option(1, "--min-depth", help="Minimum folder depth to sample"),
    max_depth: int = typer.Option(10, "--max-depth", help="Maximum folder depth to sample"),
    diversity_factor: float = typer.Option(
        0.7,
        "--diversity",
        help="Diversity factor (0-1): higher = more diverse folder names",
    ),
) -> None:
    """Create high-quality seed training set from snapshot folders.

    This command selects a diverse set of folders from a snapshot and writes them
    to a CSV file for manual labeling. The CSV includes context like parent names,
    child names, sibling names, and file extensions to aid in labeling.

    Example:
        uv run python organizer.py training select-data \\
            --sample-size 200 \\
            --min-depth 2 \\
            --max-depth 6 \\
            seed_samples.csv
    """
    typer.echo(f"Selecting training samples from snapshot...")

    # Initialize storage manager
    storage = StorageManager(storage_path)

    # Get snapshot
    with storage.get_index_session(read_only=True) as session:
        if snapshot_id is None:
            # Get most recent snapshot
            snapshot = session.execute(
                sql_select(Snapshot).order_by(Snapshot.created_at.desc())
            ).scalars().first()

            if not snapshot:
                typer.echo("❌ No snapshots found in index.db", err=True)
                typer.echo("   Run 'gather' command first to create a snapshot", err=True)
                raise typer.Exit(1)

            snapshot_id = snapshot.snapshot_id
            typer.echo(f"Using most recent snapshot: {snapshot_id} (created {snapshot.created_at})")
        else:
            # Validate snapshot exists
            snapshot = session.execute(
                sql_select(Snapshot).where(Snapshot.snapshot_id == snapshot_id)
            ).scalars().first()

            if not snapshot:
                typer.echo(f"❌ Snapshot {snapshot_id} not found in index.db", err=True)
                raise typer.Exit(1)

            typer.echo(f"Using snapshot: {snapshot_id} (created {snapshot.created_at})")

        # Select samples
        typer.echo(
            f"Selecting {sample_size} samples "
            f"(depth {min_depth}-{max_depth}, diversity={diversity_factor})..."
        )

        selected_nodes = select_training_samples(
            session=session,
            snapshot_id=snapshot_id,
            sample_size=sample_size,
            min_depth=min_depth,
            max_depth=max_depth,
            diversity_factor=diversity_factor,
        )

        if not selected_nodes:
            typer.echo(
                f"❌ No folders found in depth range {min_depth}-{max_depth}",
                err=True,
            )
            raise typer.Exit(1)

        typer.echo(f"✓ Selected {len(selected_nodes)} folders")

        # Write CSV
        typer.echo(f"Writing CSV to {output_csv}...")
        write_sample_csv(
            output_path=output_csv,
            nodes=selected_nodes,
            session=session,
            snapshot_id=snapshot_id,
        )

    typer.echo(f"✓ Done! CSV written to {output_csv}")
    typer.echo(
        "\nNext steps:\n"
        "  1. Open the CSV file and fill in the 'label' column for each row\n"
        "  2. Run 'training apply-classifications' to validate and import the labels"
    )


@training_app.command("apply-classifications")
def apply_classifications(
    input_csv: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="CSV file with manual classifications",
    ),
    storage_path: Path = typer.Option(
        None,
        "--storage",
        "-s",
        help="Storage directory (contains index.db). Defaults to data/",
    ),
    training_db_path: Path = typer.Option(
        None,
        "--training-db",
        "-t",
        help="Path to training.db. Defaults to storage_path/training.db",
    ),
    labeler: str = typer.Option("manual", "--labeler", help="Name of the labeler (e.g., 'manual', 'human-v1')"),
    split: str = typer.Option(
        None,
        "--split",
        help="Data split: 'train', 'validation', or 'test'",
    ),
    validate_only: bool = typer.Option(
        False,
        "--validate-only",
        help="Only validate CSV without writing to database",
    ),
) -> None:
    """Validate manually-labeled CSV and store in training database.

    This command reads a CSV file with manual classifications, validates the labels,
    extracts features for all nodes, and stores them in the training database.

    Example:
        # Validate only
        uv run python organizer.py training apply-classifications \\
            --validate-only \\
            seed_samples.csv

        # Apply to training database
        uv run python organizer.py training apply-classifications \\
            --labeler "human-v1" \\
            --split "train" \\
            seed_samples.csv
    """
    typer.echo(f"Reading CSV from {input_csv}...")

    # Read and parse CSV
    try:
        rows = read_classification_csv(input_csv)
    except ValueError as e:
        typer.echo(f"❌ CSV parsing error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"✓ Read {len(rows)} rows")

    # Validate labels
    typer.echo("Validating labels...")

    try:
        validate_all_labels_present(rows)
        validate_label_values(rows)
    except ValueError as e:
        typer.echo(f"❌ Validation error:\n{e}", err=True)
        raise typer.Exit(1)

    typer.echo("✓ All labels are valid")

    if validate_only:
        typer.echo("✓ Validation complete (--validate-only mode)")
        return

    # Validate split value
    if split and split not in ('train', 'validation', 'test'):
        typer.echo(
            f"❌ Invalid split value: '{split}'. Must be 'train', 'validation', or 'test'",
            err=True,
        )
        raise typer.Exit(1)

    # Initialize storage manager
    storage = StorageManager(storage_path)

    # Determine training DB path
    if training_db_path is None:
        training_db_path = storage.index_path.parent / "training.db"

    typer.echo(f"Using training database: {training_db_path}")

    # Get unique snapshot IDs from CSV
    snapshot_ids = sorted(set(row['snapshot_id'] for row in rows))
    typer.echo(f"Processing {len(snapshot_ids)} snapshot(s): {snapshot_ids}")

    # Extract features and apply labels
    training_session = get_or_create_training_session(training_db_path)

    try:
        for snapshot_id in snapshot_ids:
            typer.echo(f"\nProcessing snapshot {snapshot_id}...")

            with storage.get_index_session(read_only=True) as index_session:
                # Validate snapshot exists
                snapshot = index_session.execute(
                    sql_select(Snapshot).where(Snapshot.snapshot_id == snapshot_id)
                ).scalars().first()

                if not snapshot:
                    typer.echo(
                        f"❌ Snapshot {snapshot_id} not found in index.db",
                        err=True,
                    )
                    training_session.close()
                    raise typer.Exit(1)

                # Extract features for all nodes in snapshot
                typer.echo(f"  Extracting features for snapshot {snapshot_id}...")
                num_samples = extract_features(
                    index_session=index_session,
                    training_session=training_session,
                    snapshot_id=snapshot_id,
                    config=get_config(),
                )
                typer.echo(f"  ✓ Created {num_samples} training samples")

        # Apply labels from CSV
        typer.echo(f"\nApplying labels from CSV...")
        from storage.training_models import TrainingSample

        labeled_count = 0
        for row in rows:
            # Find the sample
            sample = training_session.query(TrainingSample).filter_by(
                snapshot_id=row['snapshot_id'],
                node_id=row['node_id'],
            ).first()

            if not sample:
                typer.echo(
                    f"⚠ Warning: Sample not found for node {row['node_id']} "
                    f"in snapshot {row['snapshot_id']} (may have been skipped)",
                    err=True,
                )
                continue

            # Update label
            sample.label = row['label']
            sample.label_confidence = 1.0
            sample.labeler = labeler
            if split:
                sample.split = split

            labeled_count += 1

        # Commit changes
        training_session.commit()
        typer.echo(f"✓ Applied {labeled_count} labels to training database")

    except Exception as e:
        training_session.rollback()
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)
    finally:
        training_session.close()

    typer.echo("\n✓ Done! Labels have been stored in training.db")
    typer.echo(
        f"  Labeled samples: {labeled_count}\n"
        f"  Labeler: {labeler}\n"
        f"  Split: {split or 'not specified'}"
    )


# FastAPI endpoints

if __name__ == "__main__":
    app()
