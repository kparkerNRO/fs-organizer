"""Fine-tuning CLI commands."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from sqlalchemy.orm import Session
from storage.training_manager import get_or_create_training_session

from .common import (
    create_model_run,
    get_db_session,
    get_highest_snapshot_id,
    get_newest_label_run_id,
    load_samples,
)
from .evaluation import evaluate_predictions, save_predictions_to_db
from .feature_extraction import extract_features as extract_fn
from .models import SetFitClassifier, ZeroShotClassifier
from .predict import predict, save_predictions_to_csv
from .sampling import (
    read_classification_csv,
    select_training_samples,
    validate_all_labels_present,
    validate_label_values,
    write_sample_csv,
)
from .taxonomy import get_labels
from .training import TrainingConfig, train_model
from utils.config import get_config

app = typer.Typer(
    name="fine_tuning",
    help="Commands for training and running ML classifiers",
    no_args_is_help=True,
)


@app.command()
def train(
    training_db: Path = typer.Option(
        ...,
        "--training-db",
        "-d",
        help="Path to training.db database",
        exists=True,
        dir_okay=False,
    ),
    label_run_id: int = typer.Option(
        None,
        "--label-run-id",
        "-l",
        help="Label run ID to use for training labels (defaults to newest)",
    ),
    taxonomy: str = typer.Option(
        "legacy",
        "--taxonomy",
        "-t",
        help="Label taxonomy to use: v1, v2, or legacy",
    ),
    output_dir: Path = typer.Option(
        "./leaf_classifier_setfit",
        "--output-dir",
        "-o",
        help="Directory to save trained model",
    ),
    model_id: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--model-id",
        "-m",
        help="Base sentence transformer model",
    ),
    num_epochs: int = typer.Option(
        6,
        "--num-epochs",
        "-e",
        help="Number of training epochs",
    ),
    test_size: float = typer.Option(
        0.2,
        "--test-size",
        help="Fraction of data to use for testing (0.0-1.0)",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
    hardneg_k: int = typer.Option(
        2,
        "--hardneg-k",
        help="Number of hard negatives to mine per anchor",
    ),
    hardneg_min_sim: float = typer.Option(
        0.25,
        "--hardneg-min-sim",
        help="Minimum similarity threshold for hard negative mining",
    ),
    hardneg_factor: int = typer.Option(
        2,
        "--hardneg-factor",
        help="Oversampling factor for hard negatives",
    ),
    hardneg_labels: str = typer.Option(
        "",
        "--hardneg-labels",
        help="Comma-separated labels to mine hard negatives for (defaults to all confusable labels in taxonomy)",
    ),
):
    """Train a SetFit classifier on labeled folder data from the database."""
    session = get_db_session(str(training_db))
    try:
        effective_label_run_id = label_run_id or get_newest_label_run_id(session)
        if effective_label_run_id is None:
            typer.echo("Error: No label runs found in database", err=True)
            raise typer.Exit(1)

        config = TrainingConfig(
            taxonomy=taxonomy,
            output_dir=output_dir,
            model_id=model_id,
            num_epochs=num_epochs,
            label_run_id=effective_label_run_id,
            test_size=test_size,
            seed=seed,
            hardneg_k=hardneg_k,
            hardneg_min_sim=hardneg_min_sim,
            hardneg_factor=hardneg_factor,
            hardneg_labels=hardneg_labels,
        )
        train_model(session, config)
    finally:
        session.close()


@app.command()
def predict_cli(
    training_db: Path = typer.Option(
        ...,
        "--training-db",
        "-d",
        help="Path to training.db database",
        exists=True,
        dir_okay=False,
    ),
    model_path: Path = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Path to fine-tuned SetFit model (required unless --use-baseline)",
    ),
    use_baseline: bool = typer.Option(
        False,
        "--use-baseline",
        help="Use baseline pre-trained model without fine-tuning",
    ),
    taxonomy: str = typer.Option(
        "legacy",
        "--taxonomy",
        "-t",
        help="Label taxonomy to use: v1, v2, or legacy",
    ),
    split: str = typer.Option(
        None,
        "--split",
        "-s",
        help="Only evaluate on specific split: train, validation, or test",
    ),
    labeled_only: bool = typer.Option(
        False,
        "--labeled-only",
        help="Only run on samples with labels (for evaluation)",
    ),
    save_predictions: bool = typer.Option(
        False,
        "--save-predictions",
        help="Save predictions to database",
    ),
    output_file: Path = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Save predictions to CSV file",
    ),
    label_run_id: int = typer.Option(
        None,
        "--label-run-id",
        "-l",
        help="Label run ID to use for training labels (defaults to newest)",
    ),
):
    """Run classifier predictions on training dataset."""
    if not use_baseline and not model_path:
        typer.echo("Error: --model-path is required unless --use-baseline is set", err=True)
        raise typer.Exit(1)

    session = get_db_session(str(training_db))
    try:
        effective_label_run_id = label_run_id or get_newest_label_run_id(session)
        if effective_label_run_id is None:
            typer.echo("Error: No label runs found in database", err=True)
            raise typer.Exit(1)

        samples = load_samples(session, split, labeled_only, effective_label_run_id)
        if not samples:
            typer.echo("No samples found. Exiting.")
            raise typer.Exit(0)

        if use_baseline:
            classifier = SetFitClassifier(taxonomy=taxonomy, use_baseline=True)
        else:
            classifier = SetFitClassifier(model_path=str(model_path), taxonomy=taxonomy)

        predictions, confidences, probabilities = predict(classifier, samples)

        y_true = [s.label for s in samples if s.label]
        y_pred_labeled = [pred for i, pred in enumerate(predictions) if samples[i].label]

        metrics = {}
        if y_true and y_pred_labeled:
            metrics = evaluate_predictions(y_true, y_pred_labeled, classifier.labels, verbose=True)

        config = {
            "use_baseline": use_baseline,
            "model_path": str(model_path) if model_path else None,
            "taxonomy": taxonomy,
            "split": split,
            "labeled_only": labeled_only,
            "metrics": metrics,
        }

        run_type_label = "baseline" if use_baseline else "fine-tuned"
        run = create_model_run(
            session,
            model_path=str(model_path) if model_path else None,
            taxonomy=taxonomy,
            use_baseline=use_baseline,
            config=config,
            run_type=run_type_label,
        )
        run.status = "completed"
        run.finished_at = datetime.now().isoformat()
        run.test_samples_count = len(samples)
        if metrics:
            run.final_val_accuracy = metrics.get("accuracy")
            run.final_val_f1 = metrics.get("macro_f1")

        session.commit()

        if save_predictions:
            save_predictions_to_db(
                session, samples, predictions, confidences, probabilities, run.run_id, split or "all"
            )

        if output_file:
            save_predictions_to_csv(output_file, samples, predictions, confidences)

    finally:
        session.close()


@app.command("zero-shot")
def zero_shot_cli(
    training_db: Path = typer.Option(
        ...,
        "--training-db",
        "-d",
        help="Path to training.db database",
        exists=True,
        dir_okay=False,
    ),
    taxonomy: str = typer.Option(
        "v2",
        "--taxonomy",
        "-t",
        help="Label taxonomy to use: v1, v2, or legacy",
    ),
    split: str = typer.Option(
        None,
        "--split",
        "-s",
        help="Only evaluate on specific split: train, validation, or test",
    ),
    labeled_only: bool = typer.Option(
        False,
        "--labeled-only",
        help="Only run on samples with labels (for evaluation)",
    ),
    save_predictions: bool = typer.Option(
        False,
        "--save-predictions",
        help="Save predictions to database",
    ),
    output_file: Path = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Save predictions to CSV file",
    ),
    label_run_id: int = typer.Option(
        None,
        "--label-run-id",
        "-l",
        help="Label run ID to use for training labels (defaults to newest)",
    ),
):
    """Run zero-shot classification using embedding similarity."""
    session = get_db_session(str(training_db))
    try:
        effective_label_run_id = label_run_id or get_newest_label_run_id(session)
        if effective_label_run_id is None:
            typer.echo("Error: No label runs found in database", err=True)
            raise typer.Exit(1)

        samples = load_samples(session, split, labeled_only, effective_label_run_id)
        if not samples:
            typer.echo("No samples found. Exiting.")
            raise typer.Exit(0)

        classifier = ZeroShotClassifier(taxonomy=taxonomy)
        predictions, confidences, probabilities = predict(classifier, samples)

        y_true = [s.label for s in samples if s.label]
        y_pred_labeled = [pred for i, pred in enumerate(predictions) if samples[i].label]

        metrics = {}
        if y_true and y_pred_labeled:
            metrics = evaluate_predictions(y_true, y_pred_labeled, classifier.labels, verbose=True)

        config = {
            "use_zero_shot": True,
            "taxonomy": taxonomy,
            "split": split,
            "labeled_only": labeled_only,
            "metrics": metrics,
        }
        run = create_model_run(
            session,
            model_path=None,
            taxonomy=taxonomy,
            use_baseline=False,
            config=config,
            run_type="zero-shot",
        )
        run.status = "completed"
        run.finished_at = datetime.now().isoformat()
        run.test_samples_count = len(samples)
        if metrics:
            run.final_val_accuracy = metrics.get("accuracy")
            run.final_val_f1 = metrics.get("macro_f1")

        session.commit()

        if save_predictions:
            save_predictions_to_db(
                session, samples, predictions, confidences, probabilities, run.run_id, split or "all"
            )

        if output_file:
            save_predictions_to_csv(output_file, samples, predictions, confidences)

    finally:
        session.close()


@app.command()
def extract_features(
    index_db: Path = typer.Option(
        ...,
        "--index-db",
        "-i",
        help="Path to index.db database",
        exists=True,
        dir_okay=False,
    ),
    training_db: Path = typer.Option(
        ...,
        "--training-db",
        "-t",
        help="Path to training.db database (will be created if doesn't exist)",
    ),
    snapshot_id: Optional[int] = typer.Option(
        None,
        "--snapshot-id",
        "-s",
        help="Snapshot ID to extract features from (defaults to highest snapshot_id if unset)",
    ),
    batch_size: int = typer.Option(
        1000,
        "--batch-size",
        "-b",
        help="Number of samples to insert per batch",
    ),
):
    """Extract features from index.db and populate training.db."""
    index_session = get_db_session(str(index_db))
    try:
        effective_snapshot_id = snapshot_id or get_highest_snapshot_id(index_session)
        if effective_snapshot_id is None:
            typer.echo(f"Error: No snapshots found in {index_db}", err=True)
            raise typer.Exit(1)
    finally:
        index_session.close()

    training_session = get_or_create_training_session(training_db)
    try:
        label_run = LabelRun(snapshot_id=effective_snapshot_id, label_source="unlabeled")
        training_session.add(label_run)
        training_session.flush()

        num_samples = extract_fn(
            index_session=get_db_session(str(index_db)),
            training_session=training_session,
            snapshot_id=effective_snapshot_id,
            config=get_config(),
            label_run=label_run,
            batch_size=batch_size,
        )
        typer.echo(f"✓ Created {num_samples} training samples")
    except Exception as e:
        typer.echo(f"Error during feature extraction: {e}", err=True)
        training_session.rollback()
        raise typer.Exit(1)
    finally:
        training_session.close()


@app.command()
def generate_samples(
    index_db: Path = typer.Option(
        ...,
        "--index-db",
        "-i",
        help="Path to index.db database",
        exists=True,
        dir_okay=False,
    ),
    output_csv: Path = typer.Option(
        ...,
        "--output-csv",
        "-o",
        help="Path to output CSV file for manual labeling",
    ),
    snapshot_id: Optional[int] = typer.Option(
        None,
        "--snapshot-id",
        "-s",
        help="Snapshot ID to generate samples from (defaults to highest snapshot_id if unset)",
    ),
    sample_size: int = typer.Option(
        800,
        "--sample-size",
        "-n",
        help="Number of samples to generate",
    ),
    min_depth: int = typer.Option(
        1,
        "--min-depth",
        help="Minimum folder depth to sample from",
    ),
    max_depth: int = typer.Option(
        10,
        "--max-depth",
        help="Maximum folder depth to sample from",
    ),
    diversity_factor: float = typer.Option(
        0.7,
        "--diversity-factor",
        help="Balance between random and diverse sampling (0-1, higher=more diverse)",
    ),
    use_heuristic: bool = typer.Option(
        True,
        "--use-heuristic/--no-heuristic",
        help="Include heuristic classifier predictions in CSV output",
    ),
    heuristic_taxonomy: str = typer.Option(
        "v2",
        "--heuristic-taxonomy",
        help="Taxonomy for heuristic classifier (v1 or v2)",
    ),
):
    """Generate training samples CSV for manual labeling."""
    session = get_db_session(str(index_db))
    try:
        effective_snapshot_id = snapshot_id or get_highest_snapshot_id(session)
        if effective_snapshot_id is None:
            typer.echo(f"Error: No snapshots found in {index_db}", err=True)
            raise typer.Exit(1)

        samples = select_training_samples(
            session=session,
            snapshot_id=effective_snapshot_id,
            sample_size=sample_size,
            min_depth=min_depth,
            max_depth=max_depth,
            diversity_factor=diversity_factor,
        )

        if not samples:
            typer.echo("Error: No samples found matching criteria", err=True)
            raise typer.Exit(1)

        write_sample_csv(
            output_path=output_csv,
            nodes=samples,
            session=session,
            snapshot_id=effective_snapshot_id,
            use_heuristic=use_heuristic,
            heuristic_taxonomy=heuristic_taxonomy,
        )
        typer.echo(f"✓ Saved samples to {output_csv}")
    finally:
        session.close()


@app.command("apply-classifications")
def apply_classifications(
    input_csv: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="CSV file with manual classifications",
    ),
    training_db_path: Path = typer.Option(
        ...,
        "--training-db",
        "-t",
        help="Path to training.db.",
    ),
    taxonomy: str = typer.Option(
        "v2",
        "--taxonomy",
        "-x",
        help="Label taxonomy to validate against: v1, v2, or legacy",
    ),
    validate_only: bool = typer.Option(
        False,
        "--validate-only",
        help="Only validate CSV without writing to database",
    ),
    split: Optional[str] = typer.Option(
        None,
        "--split",
        help="Data split: 'train', 'validation', or 'test'",
    ),
):
    """Validate manually-labeled CSV and store in training database."""
    rows = read_classification_csv(input_csv)
    valid_labels = get_labels(taxonomy)
    validate_all_labels_present(rows)
    validate_label_values(rows, valid_labels)

    if validate_only:
        typer.echo("✓ Validation complete (--validate-only mode)")
        return

    training_session = get_or_create_training_session(training_db_path)
    try:
        snapshot_ids = sorted(set(row["snapshot_id"] for row in rows))
        label_run_by_snapshot: dict[int, int] = {}
        for snapshot_id in snapshot_ids:
            label_run = LabelRun(snapshot_id=snapshot_id, label_source="manual")
            training_session.add(label_run)
            training_session.flush()
            label_run_by_snapshot[snapshot_id] = label_run.id

        labeled_count = 0
        for row in rows:
            label_run_id = label_run_by_snapshot.get(row["snapshot_id"])
            if label_run_id is None:
                continue

            sample = (
                training_session.query(TrainingSample)
                .filter_by(label_run_id=label_run_id, node_id=row["node_id"])
                .first()
            )
            if not sample:
                continue

            sample.label = row["label"]
            sample.label_confidence = 1.0
            if split:
                sample.split = split
            labeled_count += 1
        training_session.commit()
        typer.echo(f"✓ Applied {labeled_count} labels to training database")
    finally:
        training_session.close()


if __name__ == "__main__":
    app()