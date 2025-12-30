"""Fine-tuning CLI commands.

This module provides typer commands for all fine-tuning related operations:
- Training models
- Running predictions
- Generating training samples
- Managing datasets
"""

import logging
from pathlib import Path
from typing import TypeVar

import typer
from pydantic import BaseModel, ValidationError
from storage.manager import StorageManager
from utils.config import get_config


from fine_tuning.services.feature_extraction import (
    FeatureExtractionSettings,
    extract_features_from_snapshot,
)
from fine_tuning.services.predict import (
    FullPredictSettings,
    FullZeroShotSettings,
    predict_setfit,
    predict_zero_shot,
)
from fine_tuning.services.sampling import (
    ApplyClassificationsSettings,
    GenerateSamplesSettings,
    apply_sample_classifications,
    generate_sample_data,
)
from fine_tuning.services.train import FullTrainingSettings, train_model
from fine_tuning.services.common import get_highest_snapshot_id

T = TypeVar("T", bound=BaseModel)


def setup_logging() -> None:
    """Configure logging for the fine-tuning CLI."""
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()]
    )


def load_settings(settings_class: type[T], config_path: Path) -> T:
    """Load settings from a JSON file, exiting on error."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return settings_class.model_validate_json(f.read())
    except (ValidationError, FileNotFoundError) as e:
        typer.echo(f"Error loading config file {config_path}: {e}", err=True)
        raise typer.Exit(1)


# --- Typer App Initialization ---
app = typer.Typer(
    name="fine_tuning",
    help="Commands for training and running ML classifiers",
    no_args_is_help=True,
)


# --- Typer Commands ---


@app.command()
def train(
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to a JSON file with training settings.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Train a SetFit classifier using settings from a config file."""
    setup_logging()
    settings = load_settings(FullTrainingSettings, config_path)
    manager = StorageManager(settings.storage_path, initialize_training=True)
    train_model(settings, manager)


@app.command()
def predict(
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to a JSON file with prediction settings.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run classifier predictions using settings from a config file."""
    setup_logging()
    settings = load_settings(FullPredictSettings, config_path)
    manager = StorageManager(settings.storage_path, initialize_training=True)

    if not settings.use_baseline and not settings.model_path:
        typer.echo(
            "Error: --model-path is required unless --use-baseline is set", err=True
        )
        raise typer.Exit(1)

    predict_setfit(settings, manager)

    typer.echo("Done!")


@app.command("zero-shot")
def zero_shot(
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to a JSON file with zero-shot settings.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run zero-shot classification using settings from a config file."""
    setup_logging()
    settings = load_settings(FullZeroShotSettings, config_path)
    manager = StorageManager(settings.storage_path, initialize_training=True)
    predict_zero_shot(settings, manager)
    typer.echo("Done!")


@app.command()
def extract_features(
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to a JSON file with feature extraction settings.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Extract features from index.db and populate training.db."""
    setup_logging()
    settings = load_settings(FeatureExtractionSettings, config_path)
    manager = StorageManager(settings.storage_path, initialize_training=True)

    snapshot_id = settings.snapshot_id
    if snapshot_id is None:
        typer.echo("No snapshot_id provided, using highest snapshot_id...")
        snapshot_id = get_highest_snapshot_id(manager)
        typer.echo(f"Using snapshot_id: {snapshot_id}")

    typer.echo(f"Extracting features from snapshot {snapshot_id}...")

    try:
        extract_features_from_snapshot(settings, manager, get_config(), snapshot_id)
    except Exception as e:
        typer.echo(f"Error during feature extraction: {e}", err=True)
        raise typer.Exit(1)
    typer.echo("Done!")


@app.command()
def generate_samples(
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to a JSON file with sample generation settings.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Generate training samples CSV for manual labeling."""
    setup_logging()
    settings = load_settings(GenerateSamplesSettings, config_path)
    manager = StorageManager(settings.storage_path, initialize_training=True)

    snapshot_id = settings.snapshot_id
    if snapshot_id is None:
        snapshot_id = get_highest_snapshot_id(manager)
    num_samples = generate_sample_data(settings, manager, snapshot_id)

    typer.echo(f"Saved {num_samples} samples to {settings.output_csv}")


@app.command("apply-classifications")
def apply_classifications(
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to a JSON file with classification application settings.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Validate and store manually-labeled CSV in the training database."""
    setup_logging()
    settings = load_settings(ApplyClassificationsSettings, config_path)

    manager = StorageManager(settings.storage_path, initialize_training=True)
    try:
        apply_sample_classifications(settings, manager)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("Done!")


if __name__ == "__main__":
    app()
