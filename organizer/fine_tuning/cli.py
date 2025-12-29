"""Fine-tuning CLI commands.

This module provides typer commands for all fine-tuning related operations:
- Training models
- Running predictions
- Generating training samples
- Managing datasets
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer
from datasets import Dataset
from sentence_transformers.losses import BatchHardSoftMarginTripletLoss
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, func, select
from sqlalchemy import select as sql_select
from sqlalchemy.orm import Session
from storage.index_models import Snapshot
from storage.manager import StorageManager
from storage.training_manager import get_or_create_training_session
from storage.training_models import LabelRun, TrainingSample
from utils.config import get_config

from fine_tuning.feature_extraction import extract_features as extract_fn
from fine_tuning.run_classifier import (
    SetFitClassifier,
    ZeroShotClassifier,
    create_model_run,
    evaluate_predictions,
    get_newest_label_run_id,
    load_samples,
    save_predictions_to_db,
)
from fine_tuning.sampling import (
    read_classification_csv,
    select_training_samples,
    validate_all_labels_present,
    validate_label_values,
    write_sample_csv,
)
from fine_tuning.taxonomy import get_labels, convert_label, is_valid_label
from utils.text_processing import char_trigrams, jaccard_similarity

app = typer.Typer(
    name="fine_tuning",
    help="Commands for training and running ML classifiers",
    no_args_is_help=True,
)


def _get_highest_snapshot_id(index_db: Path) -> int:
    """Get the highest snapshot_id from the index database.

    Args:
        index_db: Path to the index database

    Returns:
        The highest snapshot_id

    Raises:
        typer.Exit: If no snapshots found or database is empty
    """

    engine = create_engine(f"sqlite:///{index_db}")

    with Session(engine) as session:
        result = session.execute(select(func.max(Snapshot.snapshot_id))).scalar()

        if result is None:
            typer.echo(f"Error: No snapshots found in {index_db}", err=True)
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

    # Validate taxonomy and get labels
    if taxonomy not in ["v1", "v2", "legacy"]:
        typer.echo(f"Error: Invalid taxonomy '{taxonomy}'", err=True)
        typer.echo("Valid taxonomies: v1, v2, legacy")
        raise typer.Exit(1)

    LABELS = sorted(get_labels(taxonomy))

    # Helper function for hard negative mining
    def augment_with_hard_negatives(
        train_texts: List[str],
        train_leaf_keys: List[str],
        train_labels: List[int],
        id2label: Dict[int, str],
        confusable_labels: set,
        k: int = 2,
        min_sim: float = 0.25,
        factor: int = 2,
    ) -> Tuple[List[str], List[int]]:
        tri = [char_trigrams(lk) for lk in train_leaf_keys]
        by_label: Dict[int, List[int]] = defaultdict(list)
        for i, y in enumerate(train_labels):
            by_label[y].append(i)

        extra_texts: List[str] = []
        extra_labels: List[int] = []

        for i, y in enumerate(train_labels):
            y_name = id2label[y]
            if y_name not in confusable_labels:
                continue
            if not train_leaf_keys[i]:
                continue

            scored: List[Tuple[float, int]] = []
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

    # Connect to database and load samples
    typer.echo(f"Loading training data from {training_db}...")
    engine = create_engine(f"sqlite:///{training_db}")

    with Session(engine) as session:
        # Get newest label run if not specified
        effective_label_run_id = label_run_id
        if effective_label_run_id is None:
            effective_label_run_id = get_newest_label_run_id(session)
            if effective_label_run_id is None:
                typer.echo("Error: No label runs found in database", err=True)
                raise typer.Exit(1)
            typer.echo(f"Using newest label run: {effective_label_run_id}")
        else:
            typer.echo(f"Using specified label run: {effective_label_run_id}")

        # Load labeled samples only
        samples = load_samples(
            session, split=None, labeled_only=True, label_run_id=effective_label_run_id
        )

        if not samples:
            typer.echo("Error: No labeled training samples found in database", err=True)
            raise typer.Exit(1)

        typer.echo(f"Loaded {len(samples)} labeled training samples")

        # Validate labels
        label2id: Dict[str, int] = {l: i for i, l in enumerate(LABELS)}
        id2label: Dict[int, str] = {i: l for l, i in label2id.items()}

        unknown = {s.label for s in samples if s.label and s.label not in label2id}
        if unknown:
            typer.echo(f"Error: Unknown labels found: {sorted(unknown)}", err=True)
            typer.echo(f"Valid labels: {', '.join(LABELS)}")
            raise typer.Exit(1)

        # Extract features from samples
        typer.echo("Preparing training data...")
        texts: List[str] = []
        leaf_keys: List[str] = []
        labels: List[int] = []

        for sample in samples:
            # Skip samples without labels (should not happen with labeled_only=True)
            if not sample.label:
                continue
            texts.append(sample.text)
            # Use normalized name as leaf key for hard negative mining
            leaf_keys.append(sample.name_norm)
            labels.append(label2id[sample.label])

    # Split
    typer.echo(f"Splitting data (test_size={test_size})...")
    idx = list(range(len(labels)))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels
    )

    train_texts = [texts[i] for i in train_idx]
    train_leaf_keys = [leaf_keys[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    test_ds = Dataset.from_dict(
        {"text": [texts[i] for i in test_idx], "label": [labels[i] for i in test_idx]}
    )

    # Hard-negative oversampling
    # If hardneg_labels is empty, use sensible defaults based on taxonomy
    if not hardneg_labels.strip():
        if taxonomy == "legacy":
            default_labels = ["primary_author", "secondary_author", "collection", "subject"]
        elif taxonomy == "v1":
            default_labels = ["person_or_group", "content"]
        else:  # v2
            default_labels = ["creator_or_studio", "content_subject"]
        confusable = set(default_labels)
    else:
        confusable = {s.strip() for s in hardneg_labels.split(",") if s.strip()}

    typer.echo(f"Mining hard negatives for labels: {', '.join(confusable)}...")

    extra_texts, extra_labels = augment_with_hard_negatives(
        train_texts=train_texts,
        train_leaf_keys=train_leaf_keys,
        train_labels=train_labels,
        id2label=id2label,
        confusable_labels=confusable,
        k=hardneg_k,
        min_sim=hardneg_min_sim,
        factor=hardneg_factor,
    )

    if extra_texts:
        typer.echo(f"Added {len(extra_texts)} hard negative samples")
        train_texts = train_texts + extra_texts
        train_labels = train_labels + extra_labels

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})

    # Train model
    typer.echo(f"Initializing model: {model}")
    setfit_model = SetFitModel.from_pretrained(model)

    trainer_kwargs = dict(
        model=setfit_model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        num_epochs=num_epochs,
        column_mapping={"text": "text", "label": "label"},
    )

    if not no_triplet_loss:
        typer.echo("Note: Custom triplet loss not supported in this SetFit version, using default loss")

    trainer = SetFitTrainer(**trainer_kwargs)

    typer.echo(f"\nTraining for {num_epochs} epochs...")
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
            target_names=[id2label[i] for i in range(len(LABELS))],
            digits=4,
        )
    )

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_dir))
    typer.echo(f"\n✓ Model saved to: {output_dir}")

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
    # Validation
    if not use_baseline and not model_path:
        typer.echo(
            "Error: --model-path is required unless --use-baseline is set",
            err=True,
        )
        raise typer.Exit(1)

    if taxonomy not in ["v1", "v2", "legacy"]:
        typer.echo(f"Error: Invalid taxonomy '{taxonomy}'", err=True)
        raise typer.Exit(1)

    if split and split not in ["train", "validation", "test"]:
        typer.echo(f"Error: Invalid split '{split}'", err=True)
        raise typer.Exit(1)

    # Connect to database
    typer.echo(f"Loading samples from {training_db}...")
    engine = create_engine(f"sqlite:///{training_db}")

    with Session(engine) as session:
        # Get newest label run if not specified
        effective_label_run_id = label_run_id
        if effective_label_run_id is None:
            from fine_tuning.run_classifier import get_newest_label_run_id

            effective_label_run_id = get_newest_label_run_id(session)
            if effective_label_run_id is None:
                typer.echo("Error: No label runs found in database", err=True)
                raise typer.Exit(1)
            typer.echo(f"Using newest label run: {effective_label_run_id}")
        else:
            typer.echo(f"Using specified label run: {effective_label_run_id}")

        samples = load_samples(
            session, split=split, labeled_only=labeled_only, label_run_id=effective_label_run_id
        )
        typer.echo(f"✓ Loaded {len(samples)} samples")

        if not samples:
            typer.echo("No samples found. Exiting.")
            raise typer.Exit(0)

        # Initialize classifier
        if use_baseline:
            typer.echo(f"Initializing baseline SetFit model (taxonomy={taxonomy})...")
            classifier = SetFitClassifier(taxonomy=taxonomy, use_baseline=True)
        else:
            typer.echo(f"Loading fine-tuned model from {model_path}...")
            classifier = SetFitClassifier(model_path=str(model_path), taxonomy=taxonomy)

        # Run predictions
        typer.echo("Running predictions...")
        predictions, confidences, probabilities = classifier.predict(samples)

        # Get true labels (if available)
        y_true = [s.label for s in samples if s.label]
        y_pred_labeled = [pred for i, pred in enumerate(predictions) if samples[i].label]

        # Evaluate
        if y_true and y_pred_labeled:
            typer.echo(f"\nEvaluating {len(y_true)} labeled samples...")
            metrics = evaluate_predictions(y_true, y_pred_labeled, classifier.labels, verbose=True)
        else:
            typer.echo("\nNo labeled samples found. Skipping evaluation.")
            metrics = {}

        # Always save metrics to database (even if not saving predictions)
        typer.echo("\nSaving metrics to database...")
        from datetime import datetime

        config = {
            "use_baseline": use_baseline,
            "model_path": str(model_path) if model_path else None,
            "taxonomy": taxonomy,
            "split": split,
            "labeled_only": labeled_only,
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
        run_type_label = "baseline" if use_baseline else "fine-tuned"

        run = create_model_run(
            session,
            model_path=str(model_path) if model_path else None,
            taxonomy=taxonomy,
            use_baseline=use_baseline,
            config=config,
            run_type=run_type_label,
        )

        # Update run with metrics and metadata
        run.status = "completed"
        run.finished_at = datetime.now().isoformat()
        run.test_samples_count = len(samples)

        # Save primary metrics to dedicated fields
        if metrics:
            run.final_val_accuracy = metrics.get("accuracy")
            run.final_val_f1 = metrics.get("macro_f1")
            # weighted_f1 is stored in hyperparameters_json

        # Add notes describing the run
        metrics_summary = ""
        if metrics:
            metrics_summary = f", Accuracy: {metrics.get('accuracy', 0):.4f}, Macro-F1: {metrics.get('macro_f1', 0):.4f}, Weighted-F1: {metrics.get('weighted_f1', 0):.4f}"
        run.notes = f"Run type: {run_type_label}, Taxonomy: {taxonomy}, Split: {split or 'all'}{metrics_summary}"

        session.commit()

        typer.echo(
            f"✓ Saved metrics to database (run_id={run.run_id}, type={run_type_label}, taxonomy={taxonomy})"
        )
        if metrics:
            typer.echo(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            typer.echo(f"  Macro F1: {metrics.get('macro_f1', 0):.4f}")
            typer.echo(f"  Weighted F1: {metrics.get('weighted_f1', 0):.4f}")

        # Save predictions to database (optional)
        if save_predictions:
            typer.echo("\nSaving predictions to database...")

            num_saved = save_predictions_to_db(
                session,
                samples,
                predictions,
                confidences,
                probabilities,
                run_id=run.run_id,
                prediction_type=split or "all",
            )

            typer.echo(f"✓ Saved {num_saved} predictions")

        # Save to CSV
        if output_file:
            typer.echo(f"\nSaving predictions to {output_file}...")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            import csv

            with output_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "sample_id",
                        "name",
                        "true_label",
                        "predicted_label",
                        "confidence",
                        "is_correct",
                    ]
                )

                for sample, pred, conf in zip(samples, predictions, confidences):
                    is_correct = ""
                    if sample.label:
                        is_correct = "1" if sample.label == pred else "0"

                    writer.writerow(
                        [
                            sample.sample_id,
                            sample.name_raw,
                            sample.label or "",
                            pred,
                            f"{conf:.4f}",
                            is_correct,
                        ]
                    )

            typer.echo(f"✓ Saved {len(samples)} predictions to {output_file}")

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
    # Validation
    if taxonomy not in ["v1", "v2", "legacy"]:
        typer.echo(f"Error: Invalid taxonomy '{taxonomy}'", err=True)
        raise typer.Exit(1)

    if split and split not in ["train", "validation", "test"]:
        typer.echo(f"Error: Invalid split '{split}'", err=True)
        raise typer.Exit(1)

    # Connect to database
    typer.echo(f"Loading samples from {training_db}...")
    engine = create_engine(f"sqlite:///{training_db}")

    with Session(engine) as session:
        # Get newest label run if not specified
        effective_label_run_id = label_run_id
        if effective_label_run_id is None:
            from fine_tuning.run_classifier import get_newest_label_run_id

            effective_label_run_id = get_newest_label_run_id(session)
            if effective_label_run_id is None:
                typer.echo("Error: No label runs found in database", err=True)
                raise typer.Exit(1)
            typer.echo(f"Using newest label run: {effective_label_run_id}")
        else:
            typer.echo(f"Using specified label run: {effective_label_run_id}")

        samples = load_samples(
            session, split=split, labeled_only=labeled_only, label_run_id=effective_label_run_id
        )
        typer.echo(f"✓ Loaded {len(samples)} samples")

        if not samples:
            typer.echo("No samples found. Exiting.")
            raise typer.Exit(0)

        # Initialize zero-shot classifier
        typer.echo(f"\nInitializing zero-shot classifier (taxonomy={taxonomy})...")
        classifier = ZeroShotClassifier(taxonomy=taxonomy)

        # Run predictions
        typer.echo("\nRunning zero-shot predictions...")
        predictions, confidences, probabilities = classifier.predict(samples)

        # Get true labels (if available)
        y_true = [s.label for s in samples if s.label]
        y_pred_labeled = [pred for i, pred in enumerate(predictions) if samples[i].label]

        # Evaluate
        if y_true and y_pred_labeled:
            typer.echo(f"\nEvaluating {len(y_true)} labeled samples...")
            metrics = evaluate_predictions(y_true, y_pred_labeled, classifier.labels, verbose=True)
        else:
            typer.echo("\nNo labeled samples found. Skipping evaluation.")
            metrics = {}

        # Save metrics to database
        typer.echo("\nSaving metrics to database...")
        from datetime import datetime

        config = {
            "use_zero_shot": True,
            "taxonomy": taxonomy,
            "split": split,
            "labeled_only": labeled_only,
        }

        # Add metrics to config for storage
        if metrics:
            config["metrics"] = {
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "num_samples": metrics.get("num_samples"),
            }

        run = create_model_run(
            session,
            model_path=None,
            taxonomy=taxonomy,
            use_baseline=False,  # This is different - it's zero-shot
            config=config,
            run_type="zero-shot",
        )

        # Update run with metrics and metadata
        run.status = "completed"
        run.finished_at = datetime.now().isoformat()
        run.test_samples_count = len(samples)

        # Save primary metrics to dedicated fields
        if metrics:
            run.final_val_accuracy = metrics.get("accuracy")
            run.final_val_f1 = metrics.get("macro_f1")

        # Add notes describing the run
        metrics_summary = ""
        if metrics:
            metrics_summary = f", Accuracy: {metrics.get('accuracy', 0):.4f}, Macro-F1: {metrics.get('macro_f1', 0):.4f}, Weighted-F1: {metrics.get('weighted_f1', 0):.4f}"
        run.notes = f"Run type: zero-shot, Taxonomy: {taxonomy}, Split: {split or 'all'}{metrics_summary}"

        session.commit()

        typer.echo(
            f"✓ Saved metrics to database (run_id={run.run_id}, type=zero-shot, taxonomy={taxonomy})"
        )
        if metrics:
            typer.echo(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            typer.echo(f"  Macro F1: {metrics.get('macro_f1', 0):.4f}")
            typer.echo(f"  Weighted F1: {metrics.get('weighted_f1', 0):.4f}")

        # Save predictions to database (optional)
        if save_predictions:
            typer.echo("\nSaving predictions to database...")

            num_saved = save_predictions_to_db(
                session,
                samples,
                predictions,
                confidences,
                probabilities,
                run_id=run.run_id,
                prediction_type=split or "all",
            )

            typer.echo(f"✓ Saved {num_saved} predictions")

        # Save to CSV
        if output_file:
            typer.echo(f"\nSaving predictions to {output_file}...")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            import csv

            with output_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "sample_id",
                        "name",
                        "true_label",
                        "predicted_label",
                        "confidence",
                        "is_correct",
                    ]
                )

                for sample, pred, conf in zip(samples, predictions, confidences):
                    is_correct = ""
                    if sample.label:
                        is_correct = "1" if sample.label == pred else "0"

                    writer.writerow(
                        [
                            sample.sample_id,
                            sample.name_raw,
                            sample.label or "",
                            pred,
                            f"{conf:.4f}",
                            is_correct,
                        ]
                    )

            typer.echo(f"✓ Saved {len(samples)} predictions to {output_file}")

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

    # Get highest snapshot_id if not provided
    if snapshot_id is None:
        typer.echo("No snapshot_id provided, using highest snapshot_id...")
        snapshot_id = _get_highest_snapshot_id(index_db)
        typer.echo(f"Using snapshot_id: {snapshot_id}")

    typer.echo(f"Extracting features from snapshot {snapshot_id}...")
    typer.echo(f"  Index DB: {index_db}")
    typer.echo(f"  Training DB: {training_db}")

    # Connect to databases
    index_engine = create_engine(f"sqlite:///{index_db}")
    training_session = get_or_create_training_session(training_db)

    # Load config
    config = get_config()

    try:
        with Session(index_engine) as index_session:
            typer.echo("Processing nodes...")

            # Create label run
            label_run = LabelRun(snapshot_id=snapshot_id, label_source="unlabeled")
            training_session.add(label_run)
            training_session.flush()

            num_samples = extract_fn(
                index_session=index_session,
                training_session=training_session,
                snapshot_id=snapshot_id,
                config=config,
                label_run=label_run,
                batch_size=batch_size,
            )

            typer.echo(f"✓ Created {num_samples} training samples")
            typer.echo(f"✓ Features saved to {training_db}")

    except Exception as e:
        typer.echo(f"Error during feature extraction: {e}", err=True)
        training_session.rollback()
        raise typer.Exit(1)
    finally:
        training_session.close()

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

    typer.echo(f"Connecting to database: {index_db}")
    engine = create_engine(f"sqlite:///{index_db}")

    # Get highest snapshot_id if not provided
    if snapshot_id is None:
        typer.echo("No snapshot_id provided, using highest snapshot_id...")
        snapshot_id = _get_highest_snapshot_id(index_db)
        typer.echo(f"Using snapshot_id: {snapshot_id}")

    with Session(engine) as session:
        typer.echo(f"Selecting {sample_size} diverse samples from snapshot {snapshot_id}...")
        typer.echo(f"  Depth range: {min_depth}-{max_depth}")
        typer.echo(f"  Diversity factor: {diversity_factor}")

        samples = select_training_samples(
            session=session,
            snapshot_id=snapshot_id,
            sample_size=sample_size,
            min_depth=min_depth,
            max_depth=max_depth,
            diversity_factor=diversity_factor,
        )

        if not samples:
            typer.echo("Error: No samples found matching criteria", err=True)
            raise typer.Exit(1)

        typer.echo(f"✓ Selected {len(samples)} samples")

        # Write CSV
        typer.echo(f"Writing samples to {output_csv}...")
        if use_heuristic:
            typer.echo(f"  Including heuristic predictions (taxonomy={heuristic_taxonomy})...")
        write_sample_csv(
            output_path=output_csv,
            nodes=samples,
            session=session,
            snapshot_id=snapshot_id,
            use_heuristic=use_heuristic,
            heuristic_taxonomy=heuristic_taxonomy,
        )

        typer.echo(f"✓ Saved samples to {output_csv}")
        typer.echo(
            f"\nNext steps:\n"
            f"  1. Open {output_csv} and fill in the 'label' column\n"
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
    typer.echo("Selecting training samples from snapshot...")

    # Initialize storage manager
    storage = StorageManager(storage_path)

    # Get snapshot
    with storage.get_index_session(read_only=True) as session:
        if snapshot_id is None:
            # Get most recent snapshot
            snapshot = (
                session.execute(sql_select(Snapshot).order_by(Snapshot.created_at.desc()))
                .scalars()
                .first()
            )

            if not snapshot:
                typer.echo("❌ No snapshots found in index.db", err=True)
                typer.echo("   Run 'gather' command first to create a snapshot", err=True)
                raise typer.Exit(1)

            snapshot_id = snapshot.snapshot_id
            typer.echo(f"Using most recent snapshot: {snapshot_id} (created {snapshot.created_at})")
        else:
            # Validate snapshot exists
            snapshot = (
                session.execute(sql_select(Snapshot).where(Snapshot.snapshot_id == snapshot_id))
                .scalars()
                .first()
            )

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
    typer.echo(f"Reading CSV from {input_csv}...")

    # Read and parse CSV
    try:
        rows = read_classification_csv(input_csv)
    except ValueError as e:
        typer.echo(f"❌ CSV parsing error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"✓ Read {len(rows)} rows")

    # Validate taxonomy parameter
    if taxonomy not in ["v1", "v2", "legacy"]:
        typer.echo(f"❌ Invalid taxonomy '{taxonomy}'", err=True)
        typer.echo("Valid taxonomies: v1, v2, legacy")
        raise typer.Exit(1)

    # Validate labels using specified taxonomy
    typer.echo(f"Validating labels against {taxonomy} taxonomy...")
    valid_labels = get_labels(taxonomy)

    try:
        validate_all_labels_present(rows)
        validate_label_values(rows, valid_labels)
    except ValueError as e:
        typer.echo(f"❌ Validation error:\n{e}", err=True)
        raise typer.Exit(1)

    typer.echo("✓ All labels are valid")

    if validate_only:
        typer.echo("✓ Validation complete (--validate-only mode)")
        return

    # Validate split value
    if split and split not in ("train", "validation", "test"):
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
    snapshot_ids = sorted(set(row["snapshot_id"] for row in rows))
    typer.echo(f"Processing {len(snapshot_ids)} snapshot(s): {snapshot_ids}")

    # Extract features and apply labels
    training_session = get_or_create_training_session(training_db_path)

    try:
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
                        sql_select(Snapshot).where(Snapshot.snapshot_id == snapshot_id)
                    )
                    .scalars()
                    .first()
                )

                if not snapshot:
                    typer.echo(
                        f"❌ Snapshot {snapshot_id} not found in index.db",
                        err=True,
                    )
                    training_session.close()
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

        labeled_count = 0
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


if __name__ == "__main__":
    app()
