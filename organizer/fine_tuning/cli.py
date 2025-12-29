"""Fine-tuning CLI commands.

This module provides typer commands for all fine-tuning related operations:
- Training models
- Running predictions
- Generating training samples
- Managing datasets
"""

from pathlib import Path

import typer
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import classification_report, f1_score

from fine_tuning.classifiers import SetFitClassifier, ZeroShotClassifier
from fine_tuning.services.cli_helpers import (
    FullPredictSettings,
    FullTrainingSettings,
    FullZeroShotSettings,
    apply_labels_to_samples,
    create_label_runs,
    create_samples_for_runs,
    get_highest_snapshot_id,
    load_settings,
    predict_and_evaluate,
    prepare_training_data,
    validate_input_csv,
)
from fine_tuning.services.feature_extraction import extract_features as extract_fn
from fine_tuning.services.sampling import select_training_samples, write_sample_csv
from fine_tuning.settings import (
    ApplyClassificationsSettings,
    FeatureExtractionSettings,
    GenerateSamplesSettings,
)
from storage.manager import StorageManager
from storage.training_models import LabelRun
from utils.config import get_config

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
    settings = load_settings(FullTrainingSettings, config_path)
    manager = StorageManager(settings.storage_path, initialize_training=True)
    train_ds, test_ds, id2label = prepare_training_data(settings, manager)

    typer.echo(f"Initializing model: {settings.model}")
    model = SetFitModel.from_pretrained(settings.model)

    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        num_epochs=settings.num_epochs,
        column_mapping={"text": "text", "label": "label"},
    )
    typer.echo(f"\nTraining for {settings.num_epochs} epochs...")
    trainer.train()

    typer.echo("\nEvaluating on test set...")
    y_true = test_ds["label"]
    y_pred = trainer.model.predict(test_ds["text"])
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    typer.echo(f"\n{'=' * 80}")
    typer.echo(f"Macro F1-Score: {macro_f1:.4f}")
    typer.echo(f"{'=' * 80}\n")
    typer.echo(
        classification_report(
            y_true,
            y_pred,
            target_names=[id2label[i] for i in range(len(id2label))],
            digits=4,
        )
    )

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(settings.output_dir))
    typer.echo(f"\n✓ Model saved to: {settings.output_dir}")


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
    settings = load_settings(FullPredictSettings, config_path)
    manager = StorageManager(settings.storage_path, initialize_training=True)

    if not settings.use_baseline and not settings.model_path:
        typer.echo(
            "Error: --model-path is required unless --use-baseline is set", err=True
        )
        raise typer.Exit(1)

    if settings.use_baseline:
        typer.echo(
            f"Initializing baseline SetFit model (taxonomy={settings.taxonomy})..."
        )
        classifier = SetFitClassifier(taxonomy=settings.taxonomy, use_baseline=True)
    else:
        typer.echo(f"Loading fine-tuned model from {settings.model_path}...")
        classifier = SetFitClassifier(
            model_path=str(settings.model_path), taxonomy=settings.taxonomy
        )

    predict_and_evaluate(settings, classifier, manager)
    typer.echo("\n✓ Done!")


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
    settings = load_settings(FullZeroShotSettings, config_path)
    manager = StorageManager(settings.storage_path, initialize_training=True)
    typer.echo(f"Initializing zero-shot classifier (taxonomy={settings.taxonomy})...")
    classifier = ZeroShotClassifier(taxonomy=settings.taxonomy)
    predict_and_evaluate(settings, classifier, manager)
    typer.echo("\n✓ Done!")


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
    settings = load_settings(FeatureExtractionSettings, config_path)
    manager = StorageManager(settings.storage_path, initialize_training=True)

    snapshot_id = settings.snapshot_id
    if snapshot_id is None:
        typer.echo("No snapshot_id provided, using highest snapshot_id...")
        snapshot_id = get_highest_snapshot_id(manager)
        typer.echo(f"Using snapshot_id: {snapshot_id}")

    typer.echo(f"Extracting features from snapshot {snapshot_id}...")
    with (
        manager.get_training_session() as training_session,
        manager.get_index_session(read_only=True) as index_session,
    ):
        try:
            label_run = LabelRun(snapshot_id=snapshot_id, label_source="unlabeled")
            training_session.add(label_run)
            training_session.flush()

            num_samples = extract_fn(
                index_session=index_session,
                training_session=training_session,
                snapshot_id=snapshot_id,
                config=get_config(),
                label_run=label_run,
                batch_size=settings.batch_size,
            )
            training_session.commit()
            typer.echo(
                f"✓ Created {num_samples} training samples in {manager.get_training_db_path()}"
            )
        except Exception as e:
            typer.echo(f"Error during feature extraction: {e}", err=True)
            training_session.rollback()
            raise typer.Exit(1)
    typer.echo("\n✓ Done!")


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
    settings = load_settings(GenerateSamplesSettings, config_path)
    manager = StorageManager(settings.storage_path, initialize_training=True)

    snapshot_id = settings.snapshot_id
    if snapshot_id is None:
        snapshot_id = get_highest_snapshot_id(manager)

    with manager.get_index_session(read_only=True) as session:
        samples = select_training_samples(
            session=session,
            snapshot_id=snapshot_id,
            sample_size=settings.sample_size,
            min_depth=settings.min_depth,
            max_depth=settings.max_depth,
            diversity_factor=settings.diversity_factor,
        )
        if not samples:
            typer.echo("Error: No samples found matching criteria", err=True)
            raise typer.Exit(1)

        write_sample_csv(
            output_path=settings.output_csv,
            nodes=samples,
            session=session,
            snapshot_id=snapshot_id,
            use_heuristic=settings.use_heuristic,
            heuristic_taxonomy=settings.heuristic_taxonomy,
        )
        typer.echo(f"✓ Saved {len(samples)} samples to {settings.output_csv}")


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
    settings = load_settings(ApplyClassificationsSettings, config_path)
    rows = validate_input_csv(settings.input_csv, settings.taxonomy)

    if settings.validate_only:
        typer.echo("✓ Validation complete (--validate-only mode)")
        return

    manager = StorageManager(settings.storage_path, initialize_training=True)
    try:
        with manager.get_training_session() as training_session:
            snapshot_ids = sorted(list(set(row["snapshot_id"] for row in rows)))
            label_runs = create_label_runs(training_session, snapshot_ids)

            with manager.get_index_session(read_only=True) as index_session:
                create_samples_for_runs(index_session, training_session, label_runs)

            labeled_count = apply_labels_to_samples(
                training_session, rows, label_runs, settings.split
            )
            training_session.commit()
            typer.echo(f"✓ Applied {labeled_count} labels to training database")
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("\n✓ Done!")


if __name__ == "__main__":
    app()
