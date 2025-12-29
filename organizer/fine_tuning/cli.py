"""Fine-tuning CLI commands.

This module provides typer commands for all fine-tuning related operations:
- Training models
- Running predictions
- Generating training samples
- Managing datasets
"""
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import typer
from datasets import Dataset
from pydantic import ValidationError
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from storage.index_models import Snapshot
from storage.manager import StorageManager
from storage.training_models import LabelRun, TrainingSample
from utils.config import get_config
from utils.text_processing import char_trigrams, jaccard_similarity

from fine_tuning.classifiers import SetFitClassifier, ZeroShotClassifier
from fine_tuning.evaluation import evaluate_predictions
from fine_tuning.feature_extraction import extract_features as extract_fn
from fine_tuning.sampling import (
    read_classification_csv,
    select_training_samples,
    validate_all_labels_present,
    validate_label_values,
    write_sample_csv,
)
from fine_tuning.settings import (
    ApplyClassificationsSettings,
    CommonSettings,
    FeatureExtractionSettings,
    GenerateSamplesSettings,
    PredictSettings,
    TrainSettings,
    ZeroShotSettings,
)
from fine_tuning.taxonomy import get_labels
from storage.training_manager import (
    create_model_run,
    get_newest_label_run_id,
    load_samples,
    save_predictions_to_db,
)

# --- Typer App Initialization ---
app = typer.Typer(
    name="fine_tuning",
    help="Commands for training and running ML classifiers",
    no_args_is_help=True,
)


# --- Pydantic models for combined settings ---
class FullTrainingSettings(CommonSettings, TrainSettings):
    pass


class FullPredictSettings(CommonSettings, PredictSettings):
    pass


class FullZeroShotSettings(CommonSettings, ZeroShotSettings):
    pass


# --- Core Helper Functions ---


def _load_settings(settings_class: type, config_path: Path) -> Any:
    """Load settings from a JSON file, exiting on error."""
    try:
        return settings_class.parse_file(config_path)
    except (ValidationError, FileNotFoundError) as e:
        typer.echo(f"Error loading config file {config_path}: {e}", err=True)
        raise typer.Exit(1)


def _get_highest_snapshot_id(manager: StorageManager) -> int:
    """Get the highest snapshot_id from the index database."""
    with manager.get_index_session(read_only=True) as session:
        result = session.execute(select(func.max(Snapshot.snapshot_id))).scalar()
        if result is None:
            typer.echo(f"Error: No snapshots found in {manager.get_index_db_path()}", err=True)
            raise typer.Exit(1)
        return result


def _get_effective_label_run_id(session: Session, label_run_id: Optional[int]) -> int:
    """Get the effective label run ID, defaulting to the newest if not specified."""
    if label_run_id is not None:
        typer.echo(f"Using specified label run: {label_run_id}")
        return label_run_id

    effective_label_run_id = get_newest_label_run_id(session)
    if effective_label_run_id is None:
        typer.echo("Error: No label runs found in database", err=True)
        raise typer.Exit(1)
    typer.echo(f"Using newest label run: {effective_label_run_id}")
    return effective_label_run_id


# --- ML & Data Helper Functions ---


def _save_predictions_to_csv(
    output_file: Path,
    samples: List[TrainingSample],
    predictions: List[str],
    confidences: List[float],
) -> None:
    """Save predictions to a CSV file."""
    typer.echo(f"\nSaving predictions to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "name", "true_label", "predicted_label", "confidence", "is_correct"])
        for sample, pred, conf in zip(samples, predictions, confidences):
            is_correct = "1" if sample.label and sample.label == pred else "0"
            writer.writerow([sample.sample_id, sample.name_raw, sample.label or "", pred, f"{conf:.4f}", is_correct])
    typer.echo(f"✓ Saved {len(samples)} predictions to {output_file}")


def _augment_with_hard_negatives(
    train_texts: List[str],
    train_leaf_keys: List[str],
    train_labels: List[int],
    id2label: Dict[int, str],
    confusable_labels: set,
    k: int = 2,
    min_sim: float = 0.25,
    factor: int = 2,
) -> Tuple[List[str], List[int]]:
    """Augment training data by adding hard negative samples."""
    trigrams = [char_trigrams(key) for key in train_leaf_keys]
    indices_by_label: Dict[int, List[int]] = defaultdict(list)
    for i, label_id in enumerate(train_labels):
        indices_by_label[label_id].append(i)

    extra_texts: List[str] = []
    extra_labels: List[int] = []

    for i, label_id in enumerate(train_labels):
        label_name = id2label[label_id]
        if label_name not in confusable_labels or not train_leaf_keys[i]:
            continue

        scored_candidates: List[Tuple[float, int]] = []
        for other_label_id, indices in indices_by_label.items():
            if other_label_id == label_id:
                continue
            for other_idx in indices:
                similarity = jaccard_similarity(trigrams[i], trigrams[other_idx])
                if similarity >= min_sim:
                    scored_candidates.append((similarity, other_idx))

        if not scored_candidates:
            continue

        scored_candidates.sort(reverse=True, key=lambda t: t[0])
        picked_indices = [idx for _, idx in scored_candidates[:k]]

        for _ in range(factor):
            extra_texts.append(train_texts[i])
            extra_labels.append(label_id)
            for picked_idx in picked_indices:
                extra_texts.append(train_texts[picked_idx])
                extra_labels.append(train_labels[picked_idx])

    return extra_texts, extra_labels


def _prepare_training_data(
    settings: FullTrainingSettings,
) -> Tuple[Dataset, Dataset, Dict[int, str]]:
    """Loads, processes, and splits data into training and test datasets."""
    try:
        labels_list = sorted(get_labels(settings.taxonomy))
    except ValueError:
        typer.echo(f"Error: Invalid taxonomy '{settings.taxonomy}'", err=True)
        raise typer.Exit(1)

    label2id = {label: i for i, label in enumerate(labels_list)}
    id2label = {i: label for label, i in label2id.items()}

    manager = StorageManager(settings.storage_path)
    typer.echo(f"Loading training data from {manager.get_training_db_path()}...")
    with manager.get_training_session() as session:
        effective_label_run_id = _get_effective_label_run_id(session, settings.label_run_id)
        samples = load_samples(session, labeled_only=True, label_run_id=effective_label_run_id)

    if not samples:
        typer.echo("Error: No labeled training samples found in database", err=True)
        raise typer.Exit(1)
    typer.echo(f"Loaded {len(samples)} labeled training samples")

    unknown_labels = {s.label for s in samples if s.label and s.label not in label2id}
    if unknown_labels:
        typer.echo(f"Error: Unknown labels found: {sorted(unknown_labels)}", err=True)
        typer.echo(f"Valid labels: {', '.join(labels_list)}")
        raise typer.Exit(1)

    texts = [s.text for s in samples if s.label]
    leaf_keys = [s.name_norm for s in samples if s.label]
    labels = [label2id[s.label] for s in samples if s.label]

    typer.echo(f"Splitting data (test_size={settings.test_size})...")
    indices = list(range(len(labels)))
    train_idx, test_idx = train_test_split(indices, test_size=settings.test_size, random_state=settings.seed, stratify=labels)

    train_texts = [texts[i] for i in train_idx]
    train_leaf_keys = [leaf_keys[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    test_ds = Dataset.from_dict({"text": [texts[i] for i in test_idx], "label": [labels[i] for i in test_idx]})

    if not settings.no_hard_negatives:
        confusable = {s.strip() for s in settings.hardneg_labels.split(",") if s.strip()}
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
            train_texts += extra_texts
            train_labels += extra_labels

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    return train_ds, test_ds, id2label


def _create_and_save_run_results(
    session: Session,
    settings: Union[FullPredictSettings, FullZeroShotSettings],
    classifier: Union[SetFitClassifier, ZeroShotClassifier],
    samples: List[TrainingSample],
    predictions: List[str],
    confidences: List[float],
    probabilities: List[List[float]],
    metrics: Dict[str, Any],
) -> None:
    """Creates a ModelRun, saves all results to the database and optionally to CSV."""
    run_type_label = "zero-shot" if isinstance(classifier, ZeroShotClassifier) else "fine-tuned"
    if isinstance(settings, FullPredictSettings) and settings.use_baseline:
        run_type_label = "baseline"

    typer.echo("\nSaving results to database...")
    config = settings.dict()
    if metrics:
        config["metrics"] = metrics

    model_path = getattr(settings, "model_path", None)
    run = create_model_run(
        session,
        model_path=str(model_path) if model_path else None,
        taxonomy=settings.taxonomy,
        use_baseline=getattr(settings, "use_baseline", False),
        config=config,
        run_type=run_type_label,
    )

    run.status = "completed"
    run.finished_at = datetime.now().isoformat()
    run.test_samples_count = len(samples)

    if metrics:
        run.final_val_accuracy = metrics.get("accuracy")
        run.final_val_f1 = metrics.get("macro_f1")

    metrics_summary = f", Accuracy: {metrics.get('accuracy', 0):.4f}, Macro-F1: {metrics.get('macro_f1', 0):.4f}" if metrics else ""
    run.notes = f"Run type: {run_type_label}, Taxonomy: {settings.taxonomy}, Split: {settings.split or 'all'}{metrics_summary}"

    session.commit()
    typer.echo(f"✓ Saved run metadata (run_id={run.run_id})")

    if settings.save_predictions:
        num_saved = save_predictions_to_db(
            session, samples, predictions, confidences, probabilities, run_id=run.run_id, prediction_type=settings.split or "all"
        )
        typer.echo(f"✓ Saved {num_saved} predictions to database")

    if settings.output_file:
        _save_predictions_to_csv(settings.output_file, samples, predictions, confidences)


def _predict_and_evaluate(
    settings: Union[FullPredictSettings, FullZeroShotSettings],
    classifier: Union[SetFitClassifier, ZeroShotClassifier],
) -> None:
    """Shared logic for prediction, evaluation, and saving results."""
    manager = StorageManager(settings.storage_path)
    with manager.get_training_session() as session:
        effective_label_run_id = _get_effective_label_run_id(session, settings.label_run_id)

        samples = load_samples(session, split=settings.split, labeled_only=settings.labeled_only, label_run_id=effective_label_run_id)
        typer.echo(f"✓ Loaded {len(samples)} samples from {manager.get_training_db_path()}")

        if not samples:
            typer.echo("No samples found. Exiting.")
            raise typer.Exit(0)

        typer.echo("Running predictions...")
        predictions, confidences, probabilities = classifier.predict(samples)

        y_true = [s.label for s in samples if s.label]
        y_pred_labeled = [pred for i, pred in enumerate(predictions) if samples[i].label]

        metrics = {}
        if y_true and y_pred_labeled:
            typer.echo(f"\nEvaluating {len(y_true)} labeled samples...")
            metrics = evaluate_predictions(y_true, y_pred_labeled, classifier.labels, verbose=True)
        else:
            typer.echo("\nNo labeled samples found. Skipping evaluation.")

        _create_and_save_run_results(session, settings, classifier, samples, predictions, confidences, probabilities, metrics)


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
    )
) -> None:
    """Train a SetFit classifier using settings from a config file."""
    settings = _load_settings(FullTrainingSettings, config_path)
    train_ds, test_ds, id2label = _prepare_training_data(settings)

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

    typer.echo(f"\n{ '=' * 80}")
    typer.echo(f"Macro F1-Score: {macro_f1:.4f}")
    typer.echo(f"{ '=' * 80}\n")
    typer.echo(classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(len(id2label))], digits=4))

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
    )
) -> None:
    """Run classifier predictions using settings from a config file."""
    settings = _load_settings(FullPredictSettings, config_path)

    if not settings.use_baseline and not settings.model_path:
        typer.echo("Error: --model-path is required unless --use-baseline is set", err=True)
        raise typer.Exit(1)

    if settings.use_baseline:
        typer.echo(f"Initializing baseline SetFit model (taxonomy={settings.taxonomy})...")
        classifier = SetFitClassifier(taxonomy=settings.taxonomy, use_baseline=True)
    else:
        typer.echo(f"Loading fine-tuned model from {settings.model_path}...")
        classifier = SetFitClassifier(model_path=str(settings.model_path), taxonomy=settings.taxonomy)

    _predict_and_evaluate(settings, classifier)
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
    )
) -> None:
    """Run zero-shot classification using settings from a config file."""
    settings = _load_settings(FullZeroShotSettings, config_path)
    typer.echo(f"Initializing zero-shot classifier (taxonomy={settings.taxonomy})...")
    classifier = ZeroShotClassifier(taxonomy=settings.taxonomy)
    _predict_and_evaluate(settings, classifier)
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
    )
) -> None:
    """Extract features from index.db and populate training.db."""
    settings = _load_settings(FeatureExtractionSettings, config_path)
    manager = StorageManager(settings.storage_path)

    snapshot_id = settings.snapshot_id
    if snapshot_id is None:
        typer.echo("No snapshot_id provided, using highest snapshot_id...")
        snapshot_id = _get_highest_snapshot_id(manager)
        typer.echo(f"Using snapshot_id: {snapshot_id}")

    typer.echo(f"Extracting features from snapshot {snapshot_id}...")
    with manager.get_training_session() as training_session, manager.get_index_session(read_only=True) as index_session:
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
            typer.echo(f"✓ Created {num_samples} training samples in {manager.get_training_db_path()}")
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
    )
) -> None:
    """Generate training samples CSV for manual labeling."""
    settings = _load_settings(GenerateSamplesSettings, config_path)
    manager = StorageManager(settings.storage_path)

    snapshot_id = settings.snapshot_id
    if snapshot_id is None:
        snapshot_id = _get_highest_snapshot_id(manager)

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


def _validate_input_csv(input_csv: Path, taxonomy: str) -> List[Dict[str, Any]]:
    """Read and validate the classification CSV."""
    try:
        rows = read_classification_csv(input_csv)
        valid_labels = get_labels(taxonomy)
        validate_all_labels_present(rows)
        validate_label_values(rows, valid_labels)
        typer.echo("✓ All labels are valid")
        return rows
    except ValueError as e:
        typer.echo(f"❌ Validation error:\n{e}", err=True)
        raise typer.Exit(1)


def _create_label_runs(session: Session, snapshot_ids: List[int]) -> Dict[int, LabelRun]:
    """Create new manual LabelRun entries for each snapshot."""
    label_runs = {}
    for snapshot_id in snapshot_ids:
        label_run = LabelRun(snapshot_id=snapshot_id, label_source="manual")
        session.add(label_run)
        label_runs[snapshot_id] = label_run
    session.flush()  # Assign IDs to label_run objects
    return label_runs


def _create_samples_for_runs(index_session: Session, training_session: Session, label_runs: Dict[int, LabelRun]) -> None:
    """Run feature extraction to create TrainingSample stubs for each label run."""
    config = get_config()
    for snapshot_id, label_run in label_runs.items():
        num_samples = extract_fn(
            index_session=index_session,
            training_session=training_session,
            snapshot_id=snapshot_id,
            config=config,
            label_run=label_run,
        )
        typer.echo(f"  ✓ Created {num_samples} training samples for snapshot {snapshot_id}")


def _apply_labels_to_samples(session: Session, rows: List[Dict[str, Any]], label_runs: Dict[int, LabelRun], split: Optional[str]) -> int:
    """Apply labels from CSV rows to TrainingSample objects, optimized."""
    samples_by_node_id: Dict[Tuple[int, int], TrainingSample] = {}
    for label_run in label_runs.values():
        samples_for_run = session.query(TrainingSample).filter_by(label_run_id=label_run.id).all()
        for sample in samples_for_run:
            if sample.node_id is not None:
                samples_by_node_id[(label_run.snapshot_id, sample.node_id)] = sample

    labeled_count = 0
    for row in rows:
        snapshot_id = row["snapshot_id"]
        node_id = row["node_id"]
        sample = samples_by_node_id.get((snapshot_id, node_id))

        if sample:
            sample.label = row["label"]
            sample.label_confidence = 1.0
            if split:
                sample.split = split
            labeled_count += 1
    return labeled_count


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
    settings = _load_settings(ApplyClassificationsSettings, config_path)
    rows = _validate_input_csv(settings.input_csv, settings.taxonomy)

    if settings.validate_only:
        typer.echo("✓ Validation complete (--validate-only mode)")
        return

    manager = StorageManager(settings.storage_path)
    try:
        with manager.get_training_session() as training_session:
            snapshot_ids = sorted(list(set(row["snapshot_id"] for row in rows)))
            label_runs = _create_label_runs(training_session, snapshot_ids)

            with manager.get_index_session(read_only=True) as index_session:
                _create_samples_for_runs(index_session, training_session, label_runs)

            labeled_count = _apply_labels_to_samples(training_session, rows, label_runs, settings.split)
            training_session.commit()
            typer.echo(f"✓ Applied {labeled_count} labels to training database")
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("\n✓ Done!")


if __name__ == "__main__":
    app()