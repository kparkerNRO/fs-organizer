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
from sqlalchemy import func, select
from storage.index_models import Snapshot
from storage.manager import StorageManager
from storage.training_models import LabelRun
from utils.config import get_config

from fine_tuning.services.common import logger
from fine_tuning.services.feature_extraction import (
    FeatureExtractionConfigSettings,
    extract_features_from_snapshot,
)
from fine_tuning.services.predict import (
    PredictConfigSettings,
    ZeroShotConfigSettings,
    predict_setfit,
    predict_zero_shot,
)
from fine_tuning.services.sampling import (
    ApplyClassificationsSettings,
    GenerateSamplesSettings,
    apply_sample_classifications,
    generate_sample_data,
)
from fine_tuning.services.train import TrainConfigSettings, train_model
from fine_tuning.settings import StorageSettings

T = TypeVar("T", bound=BaseModel)

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "fine_tuning"


def default_config_path(filename: str) -> Path:
    """Return the default config path for a fine-tuning command."""
    return DEFAULT_CONFIG_DIR / filename


def get_effective_label_run_id(
    manager: StorageManager, label_run_id: int | None
) -> int:
    """Get the effective label run ID, defaulting to the newest if not specified."""

    if label_run_id is not None:
        logger.info(f"Using specified label run: {label_run_id}")
        return label_run_id

    with manager.get_training_session() as session:
        result = session.execute(select(func.max(LabelRun.id))).scalar()
        if result is None:
            raise ValueError(f"No label runs found in {manager.training_path}")
        return result


def get_effective_snapshot_id(manager: StorageManager, snapshot_id: int | None) -> int:
    """Get the effective snapshot ID, defaulting to the newest if not specified."""

    if snapshot_id is not None:
        logger.info(f"Using specified snapshot: {snapshot_id}")
        return snapshot_id

    with manager.get_index_session(read_only=True) as session:
        result = session.execute(select(func.max(Snapshot.snapshot_id))).scalar()
        if result is None:
            raise ValueError(f"No snapshots found in {manager.index_path}")
        return result


def setup_logging() -> None:
    """Configure logging for the fine-tuning CLI."""
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()]
    )


def load_settings(settings_class: type[T], config_path: Path | None) -> T:
    """Load settings from a JSON file, defaulting to an empty settings object if path is None."""
    if config_path is None:
        logger.info("No config path provided, using default settings.")
        return settings_class()
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return settings_class.model_validate_json(f.read())
    except FileNotFoundError:
        logger.warning(
            f"Config file not found at {config_path}, using default settings."
        )
        return settings_class()
    except ValidationError as e:
        logger.error(f"Error validating config file {config_path}: {e}")
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
    config_path: Path | None = typer.Option(
        default_config_path("train.json"),
        "--config",
        "-c",
        help="Path to a JSON file with training configuration (hyperparameters).",
    ),
    model_path: Path | None = typer.Option(
        None,
        "--model-path",
        help="Path to save the trained model (overrides default).",
    ),
    label_run_id: int | None = typer.Option(
        None,
        "--label-run-id",
        "-l",
        help="Label run ID to use for training labels (defaults to newest).",
    ),
    resume_from: Path | None = typer.Option(
        None,
        "--resume-from",
        "-r",
        help="Path to a checkpoint to resume training from (instead of starting fresh).",
    ),
) -> None:
    """Train a SetFit classifier using hyperparameters from a config file."""
    setup_logging()
    config: TrainConfigSettings = load_settings(TrainConfigSettings, config_path)
    base_settings = StorageSettings()

    manager = StorageManager(base_settings.storage_path, initialize_training=True)

    label_run_id = get_effective_label_run_id(manager, label_run_id)

    effective_model_path = model_path or base_settings.model_path

    try:
        train_model(
            config,
            manager,
            base_settings.taxonomy,
            label_run_id,
            effective_model_path,
            resume_from=resume_from,
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("Done!")


@app.command()
def predict(
    config_path: Path | None = typer.Option(
        default_config_path("predict.json"),
        "--config",
        "-c",
        help="Path to a JSON file with prediction configuration.",
    ),
    model_path: Path | None = typer.Option(
        None,
        "--model-path",
        help="Path to load the trained model (overrides default).",
    ),
    label_run_id: int | None = typer.Option(
        None,
        "--label-run-id",
        "-l",
        help="Label run ID to use for training labels (defaults to newest).",
    ),
    split: str | None = typer.Option(
        None,
        "--split",
        help="Only evaluate on specific split: train, validation, or test.",
    ),
) -> None:
    """Run classifier predictions using configuration from a config file."""
    setup_logging()
    base_settings = StorageSettings()
    config = load_settings(PredictConfigSettings, config_path)

    effective_model_path = model_path or base_settings.model_path
    if not config.use_baseline and not effective_model_path:
        typer.echo(
            "Error: --model-path is required unless use_baseline is set in config",
            err=True,
        )
        raise typer.Exit(1)
    manager = StorageManager(base_settings.storage_path, initialize_training=True)

    label_run_id = get_effective_label_run_id(manager, label_run_id)

    try:
        predict_setfit(
            config,
            manager,
            base_settings.taxonomy,
            label_run_id,
            split,
            effective_model_path,
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("Done!")


@app.command("zero-shot")
def zero_shot(
    config_path: Path | None = typer.Option(
        default_config_path("zero_shot.json"),
        "--config",
        "-c",
        help="Path to a JSON file with zero-shot configuration.",
    ),
    label_run_id: int | None = typer.Option(
        None,
        "--label-run-id",
        "-l",
        help="Label run ID to use for training labels (defaults to newest).",
    ),
    split: str | None = typer.Option(
        None,
        "--split",
        help="Only evaluate on specific split: train, validation, or test.",
    ),
) -> None:
    """Run zero-shot classification using configuration from a config file."""
    setup_logging()

    config = load_settings(ZeroShotConfigSettings, config_path)
    base_settings = StorageSettings()

    manager = StorageManager(base_settings.storage_path, initialize_training=True)
    label_run_id = get_effective_label_run_id(manager, label_run_id)
    try:
        predict_zero_shot(config, manager, base_settings.taxonomy, label_run_id, split)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo("Done!")


@app.command()
def extract_features(
    config_path: Path | None = typer.Option(
        default_config_path("extract_features.json"),
        "--config",
        "-c",
        help="Path to a JSON file with feature extraction configuration.",
    ),
    snapshot_id: int | None = typer.Option(
        None,
        "--snapshot-id",
        help="Snapshot ID to extract features from (defaults to highest snapshot_id).",
    ),
) -> None:
    """Extract features from index.db and populate training.db."""
    setup_logging()
    config = load_settings(FeatureExtractionConfigSettings, config_path)
    base_settings = StorageSettings()
    manager = StorageManager(base_settings.storage_path, initialize_training=True)

    effective_snapshot_id = get_effective_snapshot_id(manager, snapshot_id)

    typer.echo(f"Extracting features from snapshot {effective_snapshot_id}...")

    try:
        extract_features_from_snapshot(
            config,
            manager,
            get_config(),
            effective_snapshot_id,
        )
    except Exception as e:
        typer.echo(f"Error during feature extraction: {e}", err=True)
        raise typer.Exit(1)
    typer.echo("Done!")


@app.command()
def generate_samples(
    config_path: Path | None = typer.Option(
        default_config_path("generate_samples.json"),
        "--config",
        "-c",
        help="Path to a JSON file with sample generation settings.",
    ),
    snapshot_id: int | None = typer.Option(
        None,
        "--snapshot-id",
        help="Snapshot ID to extract features from (defaults to highest snapshot_id).",
    ),
    output_csv: Path | None = typer.Option(
        None,
        "--output-csv",
        help="Path to write the samples CSV (overrides default).",
    ),
) -> None:
    """Generate training samples CSV for manual labeling."""
    setup_logging()
    base_settings = StorageSettings()
    settings = load_settings(GenerateSamplesSettings, config_path)  # type: ignore[type-var]
    manager = StorageManager(base_settings.storage_path, initialize_training=True)

    output_csv = output_csv or (base_settings.storage_path / "samples.csv")

    snapshot_id = get_effective_snapshot_id(manager, snapshot_id)
    num_samples = generate_sample_data(
        settings=settings,
        manager=manager,
        snapshot_id=snapshot_id,
        output_csv=output_csv,
    )

    typer.echo(f"Saved {num_samples} samples to {output_csv}")


@app.command("apply-classifications")
def apply_classifications(
    config_path: Path | None = typer.Option(
        default_config_path("apply_classifications.json"),
        "--config",
        "-c",
        help="Path to a JSON file with classification application settings.",
    ),
    input_csv: Path | None = typer.Option(
        None,
        "--input-csv",
        help="Path to the labeled samples CSV (overrides default).",
    ),
) -> None:
    """Validate and store manually-labeled CSV in the training database."""
    setup_logging()
    settings = load_settings(ApplyClassificationsSettings, config_path)  # type: ignore[type-var]
    base_settings = StorageSettings()
    input_csv = input_csv or (base_settings.storage_path / "samples.csv")
    manager = StorageManager(base_settings.storage_path, initialize_training=True)
    try:
        apply_sample_classifications(
            settings, manager, input_csv=input_csv, taxonomy=base_settings.taxonomy
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("Done!")


if __name__ == "__main__":
    app()
