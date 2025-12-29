"""Fine-tuning CLI commands.

This module provides typer commands for all fine-tuning related operations:
- Training models
- Running predictions
- Generating training samples
- Managing datasets
"""

from collections import defaultdict
from pathlib import Path
from typing import Optional

import typer
from datasets import Dataset
from pydantic import ValidationError
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sqlalchemy import func, select
from sqlalchemy import select as sql_select

from fine_tuning.cli_settings import (
    ApplyClassificationsSettings,
    ExtractFeaturesSettings,
    GenerateSamplesSettings,
    PredictSettings,
    SelectDataSettings,
    TrainSettings,
    ZeroShotSettings,
)
from fine_tuning.feature_extraction import extract_features as extract_fn
from fine_tuning.prediction_db import load_samples, save_predictions_to_db
from fine_tuning.prediction_utils import (
    record_model_run,
    resolve_label_run_id,
    save_predictions_csv,
)
from fine_tuning.run_classifier import (
    SetFitClassifier,
    ZeroShotClassifier,
    evaluate_predictions,
)
from fine_tuning.sampling import (
    read_classification_csv,
    select_training_samples,
    validate_all_labels_present,
    validate_label_values,
    write_sample_csv,
)
from fine_tuning.taxonomy import get_labels
from storage.index_models import Snapshot
from storage.manager import StorageManager
from storage.training_models import LabelRun, TrainingSample
from utils.config import get_config
from utils.text_processing import char_trigrams, jaccard_similarity

app = typer.Typer(
    name="fine_tuning",
    help="Commands for training and running ML classifiers",
    no_args_is_help=True,
)


def _parse_settings(settings_cls, **kwargs):
    try:
        return settings_cls(**kwargs)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(part) for part in err["loc"])
            typer.echo(f"Error: {loc}: {err['msg']}", err=True)
        raise typer.Exit(1)


def _augment_with_hard_negatives(
    train_texts: list[str],
    train_leaf_keys: list[str],
    train_labels: list[int],
    id2label: dict[int, str],
    confusable_labels: set[str],
    *,
    k: int,
    min_sim: float,
    factor: int,
) -> tuple[list[str], list[int]]:
    tri = [char_trigrams(lk) for lk in train_leaf_keys]
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, y in enumerate(train_labels):
        by_label[y].append(i)

    extra_texts: list[str] = []
    extra_labels: list[int] = []

    for i, y in enumerate(train_labels):
        y_name = id2label[y]
        if y_name not in confusable_labels:
            continue
        if not train_leaf_keys[i]:
            continue

        scored: list[tuple[float, int]] = []
        for y2, idxs in by_label.items():
            if y2 == y:
                continue
            for j in idxs:
                sim = jaccard_similarity(tri[i], tri[j])
                if sim >= min_sim:
                    scored.append((sim, j))

        if not scored:
            continue

        scored.sort(reverse=True, key=lambda t: t[0])
        picked = [j for _, j in scored[:k]]

        for _ in range(factor):
            extra_texts.append(train_texts[i])
            extra_labels.append(y)
            for j in picked:
                extra_texts.append(train_texts[j])
                extra_labels.append(train_labels[j])

    return extra_texts, extra_labels


def _get_highest_snapshot_id(storage: StorageManager) -> int:
    """Get the highest snapshot_id from the index database.

    Args:
        storage: Storage manager configured with index.db

    Returns:
        The highest snapshot_id

    Raises:
        typer.Exit: If no snapshots found or database is empty
    """

    with storage.get_index_session(read_only=True) as session:
        result = session.execute(select(func.max(Snapshot.snapshot_id))).scalar()

        if result is None:
            typer.echo(f"Error: No snapshots found in {storage.index_path}", err=True)
            raise typer.Exit(1)

        return result


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
    label_run_id: int | None = typer.Option(
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
    model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--model",
        "-m",
        help="Base sentence transformer model",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Batch size for training (must be multiple of samples_per_label)",
    ),
    num_epochs: int = typer.Option(
        6,
        "--num-epochs",
        "-e",
        help="Number of training epochs",
    ),
    learning_rate: float = typer.Option(
        2e-5,
        "--learning-rate",
        "--lr",
        help="Learning rate",
    ),
    samples_per_label: int = typer.Option(
        2,
        "--samples-per-label",
        help="Samples per label for triplet loss batching",
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
    no_triplet_loss: bool = typer.Option(
        False,
        "--no-triplet-loss",
        help="Disable triplet loss (use default SetFit loss)",
    ),
):
    """Train a SetFit classifier on labeled folder data from the database.

    This command trains a SetFit model using:
    - Hard negative mining via character trigram similarity
    - Batch-hard triplet loss for better class separation
    - Stratified train/test split

    Example:
        uv run python organizer/organizer.py model train \\
            --training-db outputs/training.db \\
            --taxonomy v2 \\
            --output-dir ./models/classifier_v1 \\
            --num-epochs 8 \\
            --batch-size 32

        # Use specific label run with legacy taxonomy
        uv run python organizer/organizer.py model train \\
            --training-db outputs/training.db \\
            --label-run-id 3 \\
            --taxonomy legacy \\
            --output-dir ./models/classifier_v1
    """

    settings = _parse_settings(
        TrainSettings,
        training_db=training_db,
        label_run_id=label_run_id,
        taxonomy=taxonomy,
        output_dir=output_dir,
        model=model,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        samples_per_label=samples_per_label,
        hardneg_k=hardneg_k,
        hardneg_min_sim=hardneg_min_sim,
        hardneg_factor=hardneg_factor,
        hardneg_labels=hardneg_labels,
        test_size=test_size,
        seed=seed,
        no_triplet_loss=no_triplet_loss,
    )

    labels = sorted(get_labels(settings.taxonomy))

    # Connect to database and load samples
    typer.echo(f"Loading training data from {settings.training_db}...")
    storage = StorageManager(
        settings.training_db.parent,
        training_path=settings.training_db,
        enable_index=False,
        enable_work=False,
    )

    with storage.get_training_session() as session:
        # Get newest label run if not specified
        try:
            effective_label_run_id = resolve_label_run_id(
                session, settings.label_run_id, log=typer.echo
            )
        except ValueError:
            raise typer.Exit(1)

        # Load labeled samples only
        samples = load_samples(
            session, split=None, labeled_only=True, label_run_id=effective_label_run_id
        )

        if not samples:
            typer.echo("Error: No labeled training samples found in database", err=True)
            raise typer.Exit(1)

        typer.echo(f"Loaded {len(samples)} labeled training samples")

        # Validate labels
        label2id: dict[str, int] = {label: i for i, label in enumerate(labels)}
        id2label: dict[int, str] = {i: label for label, i in label2id.items()}

        unknown = {s.label for s in samples if s.label and s.label not in label2id}
        if unknown:
            typer.echo(f"Error: Unknown labels found: {sorted(unknown)}", err=True)
            typer.echo(f"Valid labels: {', '.join(labels)}")
            raise typer.Exit(1)

        # Extract features from samples
        typer.echo("Preparing training data...")
        texts: list[str] = []
        leaf_keys: list[str] = []
        label_ids: list[int] = []

        for sample in samples:
            # Skip samples without labels (should not happen with labeled_only=True)
            if not sample.label:
                continue
            texts.append(sample.text)
            # Use normalized name as leaf key for hard negative mining
            leaf_keys.append(sample.name_norm)
            label_ids.append(label2id[sample.label])

    # Split
    typer.echo(f"Splitting data (test_size={settings.test_size})...")
    idx = list(range(len(label_ids)))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=settings.test_size,
        random_state=settings.seed,
        stratify=label_ids,
    )

    train_texts = [texts[i] for i in train_idx]
    train_leaf_keys = [leaf_keys[i] for i in train_idx]
    train_labels = [label_ids[i] for i in train_idx]

    test_ds = Dataset.from_dict(
        {"text": [texts[i] for i in test_idx], "label": [labels[i] for i in test_idx]}
    )

    # Hard-negative oversampling
    # If hardneg_labels is empty, use sensible defaults based on taxonomy
    if not settings.hardneg_labels.strip():
        if settings.taxonomy == "legacy":
            default_labels = [
                "primary_author",
                "secondary_author",
                "collection",
                "subject",
            ]
        elif settings.taxonomy == "v1":
            default_labels = ["person_or_group", "content"]
        else:  # v2
            default_labels = ["creator_or_studio", "content_subject"]
        confusable = set(default_labels)
    else:
        confusable = {
            s.strip() for s in settings.hardneg_labels.split(",") if s.strip()
        }

    typer.echo(f"Mining hard negatives for labels: {', '.join(confusable)}...")

    extra_texts, extra_labels = _augment_with_hard_negatives(
        train_texts=train_texts,
        train_leaf_keys=train_leaf_keys,
        train_labels=train_labels,
        id2label=id2label,
        confusable_labels=confusable,
        k=settings.hardneg_k,
        min_sim=settings.hardneg_min_sim,
        factor=settings.hardneg_factor,
    )

    if extra_texts:
        typer.echo(f"Added {len(extra_texts)} hard negative samples")
        train_texts = train_texts + extra_texts
        train_labels = train_labels + extra_labels

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})

    # Train model
    typer.echo(f"Initializing model: {settings.model}")
    setfit_model = SetFitModel.from_pretrained(settings.model)

    trainer_kwargs = dict(
        model=setfit_model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        num_epochs=settings.num_epochs,
        column_mapping={"text": "text", "label": "label"},
    )

    if not settings.no_triplet_loss:
        typer.echo(
            "Note: Custom triplet loss not supported in this SetFit version, using default loss"
        )

    trainer = SetFitTrainer(**trainer_kwargs)

    typer.echo(f"\nTraining for {settings.num_epochs} epochs...")
    trainer.train()

    # Evaluate
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
            target_names=[id2label[i] for i in range(len(labels))],
            digits=4,
        )
    )

    # Save model
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(settings.output_dir))
    typer.echo(f"\n✓ Model saved to: {settings.output_dir}")

    if extra_texts:
        typer.echo(f"✓ Hard-negative oversampling added {len(extra_texts)} samples")


@app.command()
def predict(
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
    label_run_id: int | None = typer.Option(
        None,
        "--label-run-id",
        "-l",
        help="Label run ID to use for training labels (defaults to newest)",
    ),
):
    """Run classifier predictions on training dataset.

    This command loads a SetFit model (fine-tuned or baseline) and runs predictions
    on samples from the training database. It can evaluate performance on labeled
    samples and save predictions for analysis.

    Example:
        # Run fine-tuned model on test set
        uv run python -m organizer.fine_tuning.cli predict \\
            --training-db outputs/training.db \\
            --model-path ./models/classifier_v1 \\
            --split test \\
            --labeled-only \\
            --save-predictions

        # Use baseline model
        uv run python -m organizer.fine_tuning.cli predict \\
            --training-db outputs/training.db \\
            --use-baseline \\
            --taxonomy v2

        # Use specific label run
        uv run python -m organizer.fine_tuning.cli predict \\
            --training-db outputs/training.db \\
            --model-path ./models/classifier_v1 \\
            --label-run-id 3
    """
    settings = _parse_settings(
        PredictSettings,
        training_db=training_db,
        model_path=model_path,
        use_baseline=use_baseline,
        taxonomy=taxonomy,
        split=split,
        labeled_only=labeled_only,
        save_predictions=save_predictions,
        output_file=output_file,
        label_run_id=label_run_id,
    )

    # Connect to database
    typer.echo(f"Loading samples from {settings.training_db}...")
    storage = StorageManager(
        settings.training_db.parent,
        training_path=settings.training_db,
        enable_index=False,
        enable_work=False,
    )

    with storage.get_training_session() as session:
        # Get newest label run if not specified
        try:
            effective_label_run_id = resolve_label_run_id(
                session, settings.label_run_id, log=typer.echo
            )
        except ValueError:
            raise typer.Exit(1)

        samples = load_samples(
            session,
            split=settings.split,
            labeled_only=settings.labeled_only,
            label_run_id=effective_label_run_id,
        )
        typer.echo(f"✓ Loaded {len(samples)} samples")

        if not samples:
            typer.echo("No samples found. Exiting.")
            raise typer.Exit(0)

        # Initialize classifier
        if settings.use_baseline:
            typer.echo(
                f"Initializing baseline SetFit model (taxonomy={settings.taxonomy})..."
            )
            classifier = SetFitClassifier(taxonomy=settings.taxonomy, use_baseline=True)
        else:
            typer.echo(f"Loading fine-tuned model from {settings.model_path}...")
            classifier = SetFitClassifier(
                model_path=str(settings.model_path),
                taxonomy=settings.taxonomy,
            )

        # Run predictions
        typer.echo("Running predictions...")
        predictions, confidences, probabilities = classifier.predict(samples)

        # Get true labels (if available)
        y_true = [s.label for s in samples if s.label]
        y_pred_labeled = [
            pred for i, pred in enumerate(predictions) if samples[i].label
        ]

        # Evaluate
        if y_true and y_pred_labeled:
            typer.echo(f"\nEvaluating {len(y_true)} labeled samples...")
            metrics = evaluate_predictions(
                y_true, y_pred_labeled, classifier.labels, verbose=True
            )
        else:
            typer.echo("\nNo labeled samples found. Skipping evaluation.")
            metrics = {}

        # Always save metrics to database (even if not saving predictions)
        typer.echo("\nSaving metrics to database...")
        config = {
            "use_baseline": settings.use_baseline,
            "model_path": str(settings.model_path) if settings.model_path else None,
            "taxonomy": settings.taxonomy,
            "split": settings.split,
            "labeled_only": settings.labeled_only,
        }

        # Add metrics to config for storage
        if metrics:
            config["metrics"] = {
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "num_samples": metrics.get("num_samples"),
            }

        # Determine run type for clarity
        run_type_label = "baseline" if settings.use_baseline else "fine-tuned"
        run = record_model_run(
            session,
            model_path=settings.model_path,
            taxonomy=settings.taxonomy,
            use_baseline=settings.use_baseline,
            config=config,
            metrics=metrics,
            run_type=run_type_label,
            sample_count=len(samples),
        )

        typer.echo(
            "✓ Saved metrics to database "
            f"(run_id={run.run_id}, type={run_type_label}, taxonomy={settings.taxonomy})"
        )
        if metrics:
            typer.echo(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            typer.echo(f"  Macro F1: {metrics.get('macro_f1', 0):.4f}")
            typer.echo(f"  Weighted F1: {metrics.get('weighted_f1', 0):.4f}")

        # Save predictions to database (optional)
        if settings.save_predictions:
            typer.echo("\nSaving predictions to database...")

            num_saved = save_predictions_to_db(
                session,
                samples,
                predictions,
                confidences,
                probabilities,
                run_id=run.run_id,
                prediction_type=settings.split or "all",
            )

            typer.echo(f"✓ Saved {num_saved} predictions")

        # Save to CSV
        if settings.output_file:
            typer.echo(f"\nSaving predictions to {settings.output_file}...")
            save_predictions_csv(
                settings.output_file,
                samples,
                predictions,
                confidences,
            )
            typer.echo(f"✓ Saved {len(samples)} predictions to {settings.output_file}")

    typer.echo("\n✓ Done!")


@app.command("zero-shot")
def zero_shot(
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
    label_run_id: int | None = typer.Option(
        None,
        "--label-run-id",
        "-l",
        help="Label run ID to use for training labels (defaults to newest)",
    ),
):
    """Run zero-shot classification using embedding similarity (no training needed).

    This command uses a sentence transformer model to classify folders by computing
    semantic similarity between folder features and label descriptions. No training
    is required - it works immediately on any dataset.

    Example:
        # Run zero-shot classifier on test set
        uv run python organizer/organizer.py model zero-shot \
            --training-db data/training.db \
            --split test \
            --labeled-only \
            --save-predictions

        # Run on all samples with v2 taxonomy
        uv run python organizer/organizer.py model zero-shot \
            --training-db data/training.db \
            --taxonomy v2

        # Use specific label run
        uv run python organizer/organizer.py model zero-shot \
            --training-db data/training.db \
            --label-run-id 3
    """
    settings = _parse_settings(
        ZeroShotSettings,
        training_db=training_db,
        taxonomy=taxonomy,
        split=split,
        labeled_only=labeled_only,
        save_predictions=save_predictions,
        output_file=output_file,
        label_run_id=label_run_id,
    )

    # Connect to database
    typer.echo(f"Loading samples from {settings.training_db}...")
    storage = StorageManager(
        settings.training_db.parent,
        training_path=settings.training_db,
        enable_index=False,
        enable_work=False,
    )

    with storage.get_training_session() as session:
        # Get newest label run if not specified
        try:
            effective_label_run_id = resolve_label_run_id(
                session, settings.label_run_id, log=typer.echo
            )
        except ValueError:
            raise typer.Exit(1)

        samples = load_samples(
            session,
            split=settings.split,
            labeled_only=settings.labeled_only,
            label_run_id=effective_label_run_id,
        )
        typer.echo(f"✓ Loaded {len(samples)} samples")

        if not samples:
            typer.echo("No samples found. Exiting.")
            raise typer.Exit(0)

        # Initialize zero-shot classifier
        typer.echo(
            f"\nInitializing zero-shot classifier (taxonomy={settings.taxonomy})..."
        )
        classifier = ZeroShotClassifier(taxonomy=settings.taxonomy)

        # Run predictions
        typer.echo("\nRunning zero-shot predictions...")
        predictions, confidences, probabilities = classifier.predict(samples)

        # Get true labels (if available)
        y_true = [s.label for s in samples if s.label]
        y_pred_labeled = [
            pred for i, pred in enumerate(predictions) if samples[i].label
        ]

        # Evaluate
        if y_true and y_pred_labeled:
            typer.echo(f"\nEvaluating {len(y_true)} labeled samples...")
            metrics = evaluate_predictions(
                y_true, y_pred_labeled, classifier.labels, verbose=True
            )
        else:
            typer.echo("\nNo labeled samples found. Skipping evaluation.")
            metrics = {}

        # Save metrics to database
        typer.echo("\nSaving metrics to database...")

        config = {
            "use_zero_shot": True,
            "taxonomy": settings.taxonomy,
            "split": settings.split,
            "labeled_only": settings.labeled_only,
        }

        # Add metrics to config for storage
        if metrics:
            config["metrics"] = {
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "num_samples": metrics.get("num_samples"),
            }

        run = record_model_run(
            session,
            model_path=None,
            taxonomy=settings.taxonomy,
            use_baseline=False,
            config=config,
            metrics=metrics,
            run_type="zero-shot",
            sample_count=len(samples),
        )

        typer.echo(
            f"✓ Saved metrics to database (run_id={run.run_id}, type=zero-shot, taxonomy={settings.taxonomy})"
        )
        if metrics:
            typer.echo(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            typer.echo(f"  Macro F1: {metrics.get('macro_f1', 0):.4f}")
            typer.echo(f"  Weighted F1: {metrics.get('weighted_f1', 0):.4f}")

        # Save predictions to database (optional)
        if settings.save_predictions:
            typer.echo("\nSaving predictions to database...")

            num_saved = save_predictions_to_db(
                session,
                samples,
                predictions,
                confidences,
                probabilities,
                run_id=run.run_id,
                prediction_type=settings.split or "all",
            )

            typer.echo(f"✓ Saved {num_saved} predictions")

        # Save to CSV
        if settings.output_file:
            typer.echo(f"\nSaving predictions to {settings.output_file}...")
            save_predictions_csv(
                settings.output_file,
                samples,
                predictions,
                confidences,
            )
            typer.echo(f"✓ Saved {len(samples)} predictions to {settings.output_file}")

    typer.echo("\n✓ Done!")


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
    """Extract features from index.db and populate training.db.

    This command reads all nodes from a snapshot in index.db, extracts
    classification features (hierarchical context, siblings, children, file
    extensions, etc.), and stores them in training.db for model training.

    If snapshot_id is not provided, the highest snapshot_id in the database will be used.

    Example:
        uv run python organizer/organizer.py model extract-features \\
            --index-db outputs/run/index.db \\
            --training-db outputs/training.db \\
            --snapshot-id 1
    """

    settings = _parse_settings(
        ExtractFeaturesSettings,
        index_db=index_db,
        training_db=training_db,
        snapshot_id=snapshot_id,
        batch_size=batch_size,
    )

    storage = StorageManager(
        settings.index_db.parent,
        index_path=settings.index_db,
        training_path=settings.training_db,
        enable_work=False,
    )

    # Get highest snapshot_id if not provided
    if settings.snapshot_id is None:
        typer.echo("No snapshot_id provided, using highest snapshot_id...")
        settings.snapshot_id = _get_highest_snapshot_id(storage)
        typer.echo(f"Using snapshot_id: {settings.snapshot_id}")

    typer.echo(f"Extracting features from snapshot {settings.snapshot_id}...")
    typer.echo(f"  Index DB: {settings.index_db}")
    typer.echo(f"  Training DB: {settings.training_db}")

    # Load config
    config = get_config()

    try:
        with (
            storage.get_index_session(read_only=True) as index_session,
            storage.get_training_session() as training_session,
        ):
            typer.echo("Processing nodes...")

            # Create label run
            label_run = LabelRun(
                snapshot_id=settings.snapshot_id, label_source="unlabeled"
            )
            training_session.add(label_run)
            training_session.flush()

            num_samples = extract_fn(
                index_session=index_session,
                training_session=training_session,
                snapshot_id=settings.snapshot_id,
                config=config,
                label_run=label_run,
                batch_size=settings.batch_size,
            )

            typer.echo(f"✓ Created {num_samples} training samples")
            typer.echo(f"✓ Features saved to {settings.training_db}")

    except Exception as e:
        typer.echo(f"Error during feature extraction: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("\n✓ Done! Features have been extracted to training.db")
    typer.echo(
        "\nNext steps:\n"
        "  1. Run baseline evaluation: organizer.py model predict --use-baseline\n"
        "  2. Generate labeled samples: organizer.py model generate-samples\n"
        "  3. Train model: organizer.py model train"
    )


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
    """Generate training samples CSV for manual labeling.

    This command selects diverse training samples from a snapshot using stratified
    sampling and clustering by name similarity. The output CSV can be manually
    labeled and used for training.

    If snapshot_id is not provided, the highest snapshot_id in the database will be used.

    Example:
        uv run python -m organizer.fine_tuning.cli generate-samples \\
            --index-db outputs/run/index.db \\
            --output-csv training_samples.csv \\
            --snapshot-id 1 \\
            --sample-size 1000
    """

    settings = _parse_settings(
        GenerateSamplesSettings,
        index_db=index_db,
        output_csv=output_csv,
        snapshot_id=snapshot_id,
        sample_size=sample_size,
        min_depth=min_depth,
        max_depth=max_depth,
        diversity_factor=diversity_factor,
        use_heuristic=use_heuristic,
        heuristic_taxonomy=heuristic_taxonomy,
    )

    typer.echo(f"Connecting to database: {settings.index_db}")
    storage = StorageManager(
        settings.index_db.parent,
        index_path=settings.index_db,
        enable_work=False,
        enable_training=False,
    )

    # Get highest snapshot_id if not provided
    if settings.snapshot_id is None:
        typer.echo("No snapshot_id provided, using highest snapshot_id...")
        settings.snapshot_id = _get_highest_snapshot_id(storage)
        typer.echo(f"Using snapshot_id: {settings.snapshot_id}")

    with storage.get_index_session(read_only=True) as session:
        typer.echo(
            f"Selecting {settings.sample_size} diverse samples from snapshot {settings.snapshot_id}..."
        )
        typer.echo(f"  Depth range: {settings.min_depth}-{settings.max_depth}")
        typer.echo(f"  Diversity factor: {settings.diversity_factor}")

        samples = select_training_samples(
            session=session,
            snapshot_id=settings.snapshot_id,
            sample_size=settings.sample_size,
            min_depth=settings.min_depth,
            max_depth=settings.max_depth,
            diversity_factor=settings.diversity_factor,
        )

        if not samples:
            typer.echo("Error: No samples found matching criteria", err=True)
            raise typer.Exit(1)

        typer.echo(f"✓ Selected {len(samples)} samples")

        # Write CSV
        typer.echo(f"Writing samples to {settings.output_csv}...")
        if settings.use_heuristic:
            typer.echo(
                f"  Including heuristic predictions (taxonomy={settings.heuristic_taxonomy})..."
            )
        write_sample_csv(
            output_path=settings.output_csv,
            nodes=samples,
            session=session,
            snapshot_id=settings.snapshot_id,
            use_heuristic=settings.use_heuristic,
            heuristic_taxonomy=settings.heuristic_taxonomy,
        )

        typer.echo(f"✓ Saved samples to {settings.output_csv}")
        typer.echo(
            f"\nNext steps:\n"
            f"  1. Open {settings.output_csv} and fill in the 'label' column\n"
            f"  2. Use 'train' command to train a model on the labeled data"
        )


@app.command("select-data")
def select_data(
    output_csv: Path = typer.Argument(..., help="Output CSV file path"),
    storage_path: Path = typer.Option(
        None,
        "--storage",
        "-s",
        help="Storage directory (contains index.db). Defaults to data/",
    ),
    snapshot_id: int = typer.Option(
        None,
        "--snapshot-id",
        help="Snapshot ID to sample from. Uses most recent if not specified.",
    ),
    sample_size: int = typer.Option(
        200, "--sample-size", "-n", help="Target number of samples to select"
    ),
    min_depth: int = typer.Option(
        1, "--min-depth", help="Minimum folder depth to sample"
    ),
    max_depth: int = typer.Option(
        10, "--max-depth", help="Maximum folder depth to sample"
    ),
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
    settings = _parse_settings(
        SelectDataSettings,
        output_csv=output_csv,
        storage_path=storage_path,
        snapshot_id=snapshot_id,
        sample_size=sample_size,
        min_depth=min_depth,
        max_depth=max_depth,
        diversity_factor=diversity_factor,
    )

    typer.echo("Selecting training samples from snapshot...")

    # Initialize storage manager
    storage = StorageManager(settings.storage_path)

    # Get snapshot
    with storage.get_index_session(read_only=True) as session:
        if settings.snapshot_id is None:
            # Get most recent snapshot
            snapshot = (
                session.execute(
                    sql_select(Snapshot).order_by(Snapshot.created_at.desc())
                )
                .scalars()
                .first()
            )

            if not snapshot:
                typer.echo("❌ No snapshots found in index.db", err=True)
                typer.echo(
                    "   Run 'gather' command first to create a snapshot", err=True
                )
                raise typer.Exit(1)

            settings.snapshot_id = snapshot.snapshot_id
            typer.echo(
                f"Using most recent snapshot: {settings.snapshot_id} (created {snapshot.created_at})"
            )
        else:
            # Validate snapshot exists
            snapshot = (
                session.execute(
                    sql_select(Snapshot).where(
                        Snapshot.snapshot_id == settings.snapshot_id
                    )
                )
                .scalars()
                .first()
            )

            if not snapshot:
                typer.echo(
                    f"❌ Snapshot {settings.snapshot_id} not found in index.db",
                    err=True,
                )
                raise typer.Exit(1)

            typer.echo(
                f"Using snapshot: {settings.snapshot_id} (created {snapshot.created_at})"
            )

        # Select samples
        typer.echo(
            f"Selecting {settings.sample_size} samples "
            f"(depth {settings.min_depth}-{settings.max_depth}, "
            f"diversity={settings.diversity_factor})..."
        )

        selected_nodes = select_training_samples(
            session=session,
            snapshot_id=settings.snapshot_id,
            sample_size=settings.sample_size,
            min_depth=settings.min_depth,
            max_depth=settings.max_depth,
            diversity_factor=settings.diversity_factor,
        )

        if not selected_nodes:
            typer.echo(
                f"❌ No folders found in depth range {settings.min_depth}-{settings.max_depth}",
                err=True,
            )
            raise typer.Exit(1)

        typer.echo(f"✓ Selected {len(selected_nodes)} folders")

        # Write CSV
        typer.echo(f"Writing CSV to {settings.output_csv}...")
        write_sample_csv(
            output_path=settings.output_csv,
            nodes=selected_nodes,
            session=session,
            snapshot_id=settings.snapshot_id,
        )

    typer.echo(f"✓ Done! CSV written to {settings.output_csv}")
    typer.echo(
        "\nNext steps:\n"
        "  1. Open the CSV file and fill in the 'label' column for each row\n"
        "  2. Run 'training apply-classifications' to validate and import the labels"
    )


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
    labeler: str = typer.Option(
        "manual", "--labeler", help="Name of the labeler (e.g., 'manual', 'human-v1')"
    ),
    split: str = typer.Option(
        None,
        "--split",
        help="Data split: 'train', 'validation', or 'test'",
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
    settings = _parse_settings(
        ApplyClassificationsSettings,
        input_csv=input_csv,
        storage_path=storage_path,
        training_db_path=training_db_path,
        labeler=labeler,
        split=split,
        taxonomy=taxonomy,
        validate_only=validate_only,
    )

    typer.echo(f"Reading CSV from {settings.input_csv}...")

    # Read and parse CSV
    try:
        rows = read_classification_csv(settings.input_csv)
    except ValueError as e:
        typer.echo(f"❌ CSV parsing error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"✓ Read {len(rows)} rows")

    # Validate labels using specified taxonomy
    typer.echo(f"Validating labels against {settings.taxonomy} taxonomy...")
    valid_labels = get_labels(settings.taxonomy)

    try:
        validate_all_labels_present(rows)
        validate_label_values(rows, valid_labels)
    except ValueError as e:
        typer.echo(f"❌ Validation error:\n{e}", err=True)
        raise typer.Exit(1)

    typer.echo("✓ All labels are valid")

    if settings.validate_only:
        typer.echo("✓ Validation complete (--validate-only mode)")
        return

    # Initialize storage manager
    storage = StorageManager(
        settings.storage_path,
        training_path=settings.training_db_path,
    )

    # Determine training DB path
    if settings.training_db_path is None:
        settings.training_db_path = storage.training_path

    typer.echo(f"Using training database: {settings.training_db_path}")

    # Get unique snapshot IDs from CSV
    snapshot_ids = sorted(set(row["snapshot_id"] for row in rows))
    typer.echo(f"Processing {len(snapshot_ids)} snapshot(s): {snapshot_ids}")

    labeled_count = 0

    try:
        with storage.get_training_session() as training_session:
            # Track label_run_id for each snapshot
            label_run_by_snapshot: dict[int, int] = {}

            for snapshot_id in snapshot_ids:
                # Create label run
                label_run = LabelRun(snapshot_id=snapshot_id, label_source="manual")
                training_session.add(label_run)
                training_session.flush()

                # Track the label_run_id for this snapshot
                label_run_by_snapshot[snapshot_id] = label_run.id

                typer.echo(f"\nProcessing snapshot {snapshot_id}...")

                with storage.get_index_session(read_only=True) as index_session:
                    # Validate snapshot exists
                    snapshot = (
                        index_session.execute(
                            sql_select(Snapshot).where(
                                Snapshot.snapshot_id == snapshot_id
                            )
                        )
                        .scalars()
                        .first()
                    )

                    if not snapshot:
                        typer.echo(
                            f"❌ Snapshot {snapshot_id} not found in index.db",
                            err=True,
                        )
                        raise typer.Exit(1)

                    # Extract features for all nodes in snapshot
                    typer.echo(f"  Extracting features for snapshot {snapshot_id}...")
                    num_samples = extract_fn(
                        index_session=index_session,
                        training_session=training_session,
                        snapshot_id=snapshot_id,
                        config=get_config(),
                        label_run=label_run,
                    )
                    typer.echo(f"  ✓ Created {num_samples} training samples")

            # Apply labels from CSV
            typer.echo("\nApplying labels from CSV...")

            for row in rows:
                # Get the label_run_id for this snapshot
                label_run_id = label_run_by_snapshot.get(row["snapshot_id"])
                if label_run_id is None:
                    typer.echo(
                        f"⚠ Warning: No label run found for snapshot {row['snapshot_id']}",
                        err=True,
                    )
                    continue

                # Find the sample for this label run
                sample = (
                    training_session.query(TrainingSample)
                    .filter_by(
                        label_run_id=label_run_id,
                        node_id=row["node_id"],
                    )
                    .first()
                )

                if not sample:
                    typer.echo(
                        f"⚠ Warning: Sample not found for node {row['node_id']} "
                        f"in snapshot {row['snapshot_id']} (may have been skipped)",
                        err=True,
                    )
                    continue

                # Update label
                sample.label = row["label"]
                sample.label_confidence = 1.0
                if settings.split:
                    sample.split = settings.split

                labeled_count += 1

            # Commit changes
            training_session.commit()
            typer.echo(f"✓ Applied {labeled_count} labels to training database")

    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("\n✓ Done! Labels have been stored in training.db")
    typer.echo(
        f"  Labeled samples: {labeled_count}\n"
        f"  Labeler: {settings.labeler}\n"
        f"  Split: {settings.split or 'not specified'}"
    )


if __name__ == "__main__":
    app()
