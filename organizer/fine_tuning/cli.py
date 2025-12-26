"""Fine-tuning CLI commands.

This module provides typer commands for all fine-tuning related operations:
- Training models
- Running predictions
- Generating training samples
- Managing datasets
"""

import typer
from pathlib import Path
from typing import Optional

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
    from sqlalchemy import create_engine, func, select
    from sqlalchemy.orm import Session
    from storage.index_models import Snapshot

    engine = create_engine(f"sqlite:///{index_db}")

    with Session(engine) as session:
        result = session.execute(
            select(func.max(Snapshot.snapshot_id))
        ).scalar()

        if result is None:
            typer.echo(f"Error: No snapshots found in {index_db}", err=True)
            raise typer.Exit(1)

        return result


@app.command()
def train(
    data: Path = typer.Option(
        ...,
        "--data",
        "-d",
        help="CSV file with headers: path,label",
        exists=True,
        dir_okay=False,
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
        "primary_author,secondary_author,collection,subject",
        "--hardneg-labels",
        help="Comma-separated labels to mine hard negatives for",
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
    """Train a SetFit classifier on labeled folder data.

    This command trains a SetFit model using:
    - Hard negative mining via character trigram similarity
    - Batch-hard triplet loss for better class separation
    - Stratified train/test split

    Example:
        uv run python -m organizer.fine_tuning.cli train \\
            --data training_samples.csv \\
            --output-dir ./models/classifier_v1 \\
            --num-epochs 8 \\
            --batch-size 32
    """
    from collections import defaultdict
    from typing import Dict, List, Tuple
    import csv
    import re

    from datasets import Dataset
    from setfit import SetFitModel, SetFitTrainer
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import train_test_split
    from sentence_transformers.losses import BatchHardSoftMarginTripletLoss

    # Label definitions
    LABELS = [
        "primary_author",
        "secondary_author",
        "collection",
        "subject",
        "media_format",
        "media_type",
        "variant",
        "other",
    ]

    # Validation
    if not no_triplet_loss and (batch_size % samples_per_label != 0):
        typer.echo(
            f"Error: batch_size ({batch_size}) must be a multiple of "
            f"samples_per_label ({samples_per_label}) when using triplet loss",
            err=True,
        )
        raise typer.Exit(1)

    # Helper functions
    _SPLIT_RE = re.compile(r"[\\/._\- ]+")
    _CAMEL_RE_1 = re.compile(r"([a-z0-9])([A-Z])")
    _CAMEL_RE_2 = re.compile(r"([A-Z]+)([A-Z][a-z])")

    def _tokenize_segment(seg: str) -> List[str]:
        seg = seg.strip()
        if not seg:
            return []
        seg = _CAMEL_RE_2.sub(r"\1 \2", seg)
        seg = _CAMEL_RE_1.sub(r"\1 \2", seg)
        toks = [t for t in _SPLIT_RE.split(seg) if t]
        return [t.lower() for t in toks]

    def split_path_parts(path: str) -> List[str]:
        p = path.strip().replace("\\", "/")
        return [s for s in p.split("/") if s.strip()]

    def path_to_leaf_example(path: str) -> Tuple[str, str]:
        parts = split_path_parts(path)
        if not parts:
            return "depth:0 parents: (root) || leaf: (empty)", ""

        leaf = parts[-1]
        parents = parts[:-1]

        parent_toks: List[str] = []
        for seg in parents:
            parent_toks.extend(_tokenize_segment(seg))

        leaf_toks = _tokenize_segment(leaf)

        depth_tok = f"depth:{len(parts)}"
        parents_text = " ".join(parent_toks) if parent_toks else "(root)"
        leaf_text = " ".join(leaf_toks) if leaf_toks else "(empty)"

        text = f"{depth_tok} parents: {parents_text} || leaf: {leaf_text}"
        leaf_key = " ".join(leaf_toks)
        return text, leaf_key

    def char_trigrams(s: str) -> set:
        s = re.sub(r"\s+", " ", s.strip())
        if len(s) < 3:
            return {s} if s else set()
        return {s[i : i + 3] for i in range(len(s) - 2)}

    def jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

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
                    sim = jaccard(tri[i], tri[j])
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

    # Read CSV
    typer.echo(f"Reading training data from {data}...")
    rows: List[Tuple[str, str]] = []
    with open(data, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            typer.echo("Error: CSV must have headers: path,label", err=True)
            raise typer.Exit(1)

        for r in reader:
            p = (r.get("path") or "").strip()
            y = (r.get("label") or "").strip()
            if p and y:
                rows.append((p, y))

    if not rows:
        typer.echo("Error: No training rows found in CSV", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(rows)} training samples")

    # Validate labels
    label2id: Dict[str, int] = {l: i for i, l in enumerate(LABELS)}
    id2label: Dict[int, str] = {i: l for l, i in label2id.items()}

    unknown = {y for _, y in rows if y not in label2id}
    if unknown:
        typer.echo(f"Error: Unknown labels found: {sorted(unknown)}", err=True)
        typer.echo(f"Valid labels: {', '.join(LABELS)}")
        raise typer.Exit(1)

    # Convert to features
    typer.echo("Extracting features...")
    texts: List[str] = []
    leaf_keys: List[str] = []
    labels: List[int] = []

    for p, y in rows:
        t, lk = path_to_leaf_example(p)
        texts.append(t)
        leaf_keys.append(lk)
        labels.append(label2id[y])

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

    # Validate triplet loss requirements
    if not no_triplet_loss:
        counts = defaultdict(int)
        for y in train_labels:
            counts[y] += 1
        too_small = [id2label[y] for y, c in counts.items() if c < 2]
        if too_small:
            typer.echo(
                f"Error: Triplet loss requires >=2 train examples per label.\n"
                f"These labels have <2 in the train split: {too_small}",
                err=True,
            )
            raise typer.Exit(1)

    # Hard-negative oversampling
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
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        column_mapping={"text": "text", "label": "label"},
    )

    if not no_triplet_loss:
        typer.echo("Using batch-hard triplet loss")
        trainer_kwargs.update(
            loss=BatchHardSoftMarginTripletLoss,
            samples_per_label=samples_per_label,
        )

    trainer = SetFitTrainer(**trainer_kwargs)

    typer.echo(f"\nTraining for {num_epochs} epochs...")
    trainer.train()

    # Evaluate
    typer.echo("\nEvaluating on test set...")
    y_true = test_ds["label"]
    y_pred = trainer.model.predict(test_ds["text"])
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    typer.echo(f"\n{'='*80}")
    typer.echo(f"Macro F1-Score: {macro_f1:.4f}")
    typer.echo(f"{'='*80}\n")
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

    # Import and run
    from organizer.fine_tuning.run_classifier import (
        SetFitClassifier,
        create_model_run,
        evaluate_predictions,
        load_samples,
        save_predictions_to_db,
    )
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    # Connect to database
    typer.echo(f"Loading samples from {training_db}...")
    engine = create_engine(f"sqlite:///{training_db}")

    with Session(engine) as session:
        samples = load_samples(session, split=split, labeled_only=labeled_only)
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
            metrics = evaluate_predictions(
                y_true, y_pred_labeled, classifier.labels, verbose=True
            )
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

        typer.echo(f"✓ Saved metrics to database (run_id={run.run_id}, type={run_type_label}, taxonomy={taxonomy})")
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
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from organizer.fine_tuning.feature_extraction import extract_features as extract_fn
    from organizer.storage.training_manager import get_or_create_training_session
    from organizer.utils.config import get_config

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

            num_samples = extract_fn(
                index_session=index_session,
                training_session=training_session,
                snapshot_id=snapshot_id,
                config=config,
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
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from organizer.fine_tuning.training_utils import (
        select_training_samples,
        write_sample_csv,
    )

    typer.echo(f"Connecting to database: {index_db}")
    engine = create_engine(f"sqlite:///{index_db}")

    # Get highest snapshot_id if not provided
    if snapshot_id is None:
        typer.echo("No snapshot_id provided, using highest snapshot_id...")
        snapshot_id = _get_highest_snapshot_id(index_db)
        typer.echo(f"Using snapshot_id: {snapshot_id}")

    with Session(engine) as session:
        typer.echo(
            f"Selecting {sample_size} diverse samples from snapshot {snapshot_id}..."
        )
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
        write_sample_csv(
            output_path=output_csv,
            nodes=samples,
            session=session,
            snapshot_id=snapshot_id,
        )

        typer.echo(f"✓ Saved samples to {output_csv}")
        typer.echo(
            f"\nNext steps:\n"
            f"  1. Open {output_csv} and fill in the 'label' column\n"
            f"  2. Use 'train' command to train a model on the labeled data"
        )


if __name__ == "__main__":
    app()
