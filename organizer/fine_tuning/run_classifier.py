"""Run ML classifier predictions on a dataset.

This script runs a fine-tuned SetFit model (trained with leaf_classifier.py) against
a dataset and evaluates performance.

Usage:
    # Run fine-tuned model on all samples
    uv run python run_classifier.py --training_db path/to/training.db --model_path ./model_checkpoint

    # Run with specific label taxonomy (v1 or v2)
    uv run python run_classifier.py --training_db path/to/training.db --model_path ./model --taxonomy v2

    # Run on specific split
    uv run python run_classifier.py --training_db path/to/training.db --model_path ./model --split test

    # Save predictions to database
    uv run python run_classifier.py --training_db path/to/training.db --model_path ./model --save_predictions

    # Export predictions to CSV
    uv run python run_classifier.py --training_db path/to/training.db --model_path ./model --output_file predictions.csv

    # Use baseline pre-trained model (no fine-tuning)
    uv run python run_classifier.py --training_db path/to/training.db --use_baseline --taxonomy v2
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from storage.training_models import (
    SamplePrediction,
    ModelRun,
    TrainingSample,
    LabelRun,
)
from fine_tuning.taxonomy import get_labels


class ZeroShotClassifier:
    """Zero-shot classifier using embedding similarity (no training needed)."""

    def __init__(self, taxonomy: str = "v2"):
        """Initialize zero-shot classifier.

        Args:
            taxonomy: Which label set to use ('v1', 'v2', or 'legacy')
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )

        self.taxonomy = taxonomy
        self.labels = get_labels(taxonomy)

        # Load sentence transformer model
        print("Loading sentence transformer: sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Define label descriptions for semantic matching
        self.label_descriptions = self._get_label_descriptions(taxonomy)

        # Pre-compute label embeddings
        print(f"Computing embeddings for {len(self.label_descriptions)} labels...")
        self.label_names = list(self.label_descriptions.keys())
        self.label_embeddings = self.model.encode(
            list(self.label_descriptions.values()), show_progress_bar=False
        )

    def _get_label_descriptions(self, taxonomy: str) -> Dict[str, str]:
        """Get semantic descriptions for each label in the taxonomy.

        Args:
            taxonomy: Taxonomy name ('v1', 'v2', or 'legacy')

        Returns:
            Dict mapping label names to semantic descriptions
        """
        if taxonomy == "v1":
            return {
                "person_or_group": "creator publisher studio author artist cartographer person group organization company",
                "content": "location subject matter place setting scene environment theme topic content area",
                "media_bucket": "media type file format asset type maps tokens music images videos audio handouts",
                "descriptor": "variant style modifier season weather time lighting effect version alternate",
                "other": "organizational administrative folder rewards tiers bonus instructions guide notes",
                "unknown": "unclear ambiguous uncertain unclassifiable miscellaneous",
            }
        elif taxonomy == "v2":
            return {
                "creator_or_studio": "creator publisher studio author artist cartographer person group organization company",
                "content_subject": "location subject matter place setting scene environment theme topic content specific area castle tavern forest",
                "asset_type": "media type file format asset category maps tokens music images videos audio handouts illustrations",
                "descriptor": "variant style modifier season weather time lighting day night gridded effect version alternate",
                "other": "organizational administrative folder rewards tiers bonus instructions guide notes year month",
                "unknown": "unclear ambiguous uncertain unclassifiable miscellaneous",
            }
        else:  # legacy
            return {
                "primary_author": "main creator primary author original artist cartographer publisher",
                "secondary_author": "collaborator featured creator secondary author co-creator collaboration",
                "collection": "collection series set group pack bundle compilation",
                "subject": "subject matter topic theme location specific content castle tavern dungeon",
                "media_format": "file format extension type jpg png pdf mp3 wav webm vtt",
                "media_type": "media category asset type maps tokens music audio images videos handouts",
                "variant": "variant alternate version style modifier day night gridded season weather",
                "other": "organizational administrative rewards tiers bonus year month instructions notes",
            }

    def predict(
        self, samples: List[TrainingSample]
    ) -> Tuple[List[str], List[float], List[Dict[str, float]]]:
        """Predict labels using zero-shot embedding similarity.

        Args:
            samples: List of TrainingSample objects

        Returns:
            Tuple of (predictions, confidences, probabilities)
        """
        from sklearn.metrics.pairwise import cosine_similarity

        # Extract text features from samples
        texts = [sample.text for sample in samples]

        # Encode sample texts
        text_embeddings = self.model.encode(texts, show_progress_bar=True)

        # Compute cosine similarity between each sample and each label
        similarities = cosine_similarity(text_embeddings, self.label_embeddings)

        # Get predictions and confidences
        predictions = []
        confidences = []
        all_probabilities = []

        for sim_row in similarities:
            # Find best matching label
            best_idx = int(np.argmax(sim_row))
            predicted_label = self.label_names[best_idx]
            confidence = float(sim_row[best_idx])

            predictions.append(predicted_label)
            confidences.append(confidence)

            # Convert similarities to probability-like scores (softmax)
            # Use temperature scaling to make the distribution sharper
            temperature = 0.5
            exp_sim = np.exp(sim_row / temperature)
            probs = exp_sim / np.sum(exp_sim)

            prob_dict = {label: float(prob) for label, prob in zip(self.label_names, probs)}
            all_probabilities.append(prob_dict)

        return predictions, confidences, all_probabilities


class SetFitClassifier:
    """Wrapper for SetFit model (fine-tuned or baseline)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        taxonomy: str = "legacy",
        use_baseline: bool = False,
    ):
        """Initialize SetFit classifier.

        Args:
            model_path: Path to saved fine-tuned SetFit model (required unless use_baseline=True)
            taxonomy: Which label set to use ('v1', 'v2', or 'legacy')
            use_baseline: Use baseline pre-trained model without fine-tuning
        """
        try:
            from setfit import SetFitModel
        except ImportError:
            raise ImportError("SetFit not installed. Install with: pip install setfit")

        if use_baseline:
            # Use baseline pre-trained sentence transformer
            print("Loading baseline model: sentence-transformers/all-MiniLM-L6-v2")
            self.model = SetFitModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            if not model_path:
                raise ValueError("model_path required when use_baseline=False")
            self.model = SetFitModel.from_pretrained(model_path)

        self.taxonomy = taxonomy
        self.use_baseline = use_baseline

        # Get label set for taxonomy
        self.labels = get_labels(taxonomy)

    def predict(
        self, samples: List[TrainingSample]
    ) -> Tuple[List[str], List[float], List[Dict[str, float]]]:
        """Predict labels for multiple samples.

        Args:
            samples: List of TrainingSample objects

        Returns:
            Tuple of (predictions, confidences, probabilities)
        """
        # Extract text features
        texts = [sample.text for sample in samples]

        # Get predictions
        predictions = self.model.predict(texts)

        # Get probabilities
        try:
            probs = self.model.predict_proba(texts)
            confidences = np.max(probs, axis=1).tolist()

            # Convert to list of dicts
            label_list = sorted(self.labels)
            all_probabilities = []
            for prob_row in probs:
                prob_dict = {
                    label: float(prob) for label, prob in zip(label_list, prob_row)
                }
                all_probabilities.append(prob_dict)
        except Exception:
            # Fallback if predict_proba not available
            confidences = [1.0] * len(predictions)
            all_probabilities = [
                {label: (1.0 if label == pred else 0.0) for label in self.labels}
                for pred in predictions
            ]

        return predictions.tolist(), confidences, all_probabilities


def load_samples(
    session: Session,
    split: Optional[str] = None,
    labeled_only: bool = False,
    label_run_id: Optional[int] = None,
) -> List[TrainingSample]:
    """Load training samples from database.

    Args:
        session: SQLAlchemy session
        split: Optional split filter ('train', 'validation', 'test')
        labeled_only: Only load samples with labels
        label_run_id: Optional label run ID to filter by

    Returns:
        List of TrainingSample objects
    """
    query = select(TrainingSample)

    if label_run_id is not None:
        query = query.where(TrainingSample.label_run_id == label_run_id)

    if split:
        query = query.where(TrainingSample.split == split)

    if labeled_only:
        query = query.where(TrainingSample.label.isnot(None))
        query = query.where(TrainingSample.label != "")

    samples = session.execute(query).scalars().all()
    return list(samples)


def get_newest_label_run_id(session: Session) -> Optional[int]:
    """Get the newest (highest ID) label run from the database.

    Args:
        session: SQLAlchemy session

    Returns:
        The ID of the newest label run, or None if no label runs exist
    """
    result = session.execute(
        select(LabelRun.id).order_by(LabelRun.id.desc()).limit(1)
    ).scalar()
    return result


def evaluate_predictions(
    y_true: List[str],
    y_pred: List[str],
    labels: Set[str],
    verbose: bool = True,
) -> Dict:
    """Evaluate predictions and print metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Set of valid labels
        verbose: Print detailed metrics

    Returns:
        Dictionary of metrics
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=sorted(labels))
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=sorted(labels))

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_samples": len(y_true),
    }

    if verbose:
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"\nTotal Samples: {len(y_true)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1-Score: {macro_f1:.4f}")
        print(f"Weighted F1-Score: {weighted_f1:.4f}")

        print("\n" + "-" * 80)
        print("CLASSIFICATION REPORT")
        print("-" * 80)
        print(
            classification_report(
                y_true,
                y_pred,
                labels=sorted(labels),
                target_names=sorted(labels),
                digits=4,
            )
        )

        print("\n" + "-" * 80)
        print("CONFUSION MATRIX")
        print("-" * 80)
        cm = confusion_matrix(y_true, y_pred, labels=sorted(labels))
        label_list = sorted(labels)

        # Print header
        print(f"{'':20s}", end="")
        for label in label_list:
            print(f"{label[:15]:>15s}", end=" ")
        print()

        # Print rows
        for i, true_label in enumerate(label_list):
            print(f"{true_label[:20]:20s}", end="")
            for j in range(len(label_list)):
                print(f"{cm[i, j]:>15d}", end=" ")
            print()

        # Print most common errors
        print("\n" + "-" * 80)
        print("MOST COMMON ERRORS")
        print("-" * 80)

        errors = defaultdict(int)
        for true, pred in zip(y_true, y_pred):
            if true != pred:
                errors[(true, pred)] += 1

        sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)
        for (true, pred), count in sorted_errors[:10]:
            print(f"  {true:20s} -> {pred:20s}: {count:4d} errors")

    return metrics


def save_predictions_to_db(
    session: Session,
    samples: List[TrainingSample],
    predictions: List[str],
    confidences: List[float],
    probabilities: List[Dict[str, float]],
    run_id: int,
    prediction_type: str = "test",
) -> int:
    """Save predictions to database.

    Args:
        session: SQLAlchemy session
        samples: List of samples
        predictions: List of predicted labels
        confidences: List of confidence scores
        probabilities: List of probability dicts
        run_id: TrainingRun ID
        prediction_type: Type of prediction ('train', 'validation', 'test')

    Returns:
        Number of predictions saved
    """
    prediction_objects = []

    for sample, pred, conf, probs in zip(
        samples, predictions, confidences, probabilities
    ):
        is_correct = None
        if sample.label:
            is_correct = sample.label == pred

        prediction_obj = SamplePrediction(
            run_id=run_id,
            sample_id=sample.sample_id,
            predicted_label=pred,
            confidence=conf,
            probabilities_json=json.dumps(probs),
            true_label=sample.label,
            is_correct=is_correct,
            prediction_type=prediction_type,
        )
        prediction_objects.append(prediction_obj)

    session.add_all(prediction_objects)
    session.commit()

    return len(prediction_objects)


def create_model_run(
    session: Session,
    model_path: Optional[str],
    taxonomy: str,
    use_baseline: bool,
    config: Dict,
    run_type: Optional[str] = None,
    training_data_source: Optional[str] = None,
) -> ModelRun:
    """Create a new model run record.

    Args:
        session: SQLAlchemy session
        model_path: Path to fine-tuned model (or None if baseline)
        taxonomy: Taxonomy version
        use_baseline: Whether using baseline model
        config: Configuration dict
        run_type: Type of run ('training', 'evaluation', 'baseline'). Auto-detected if None.
        training_data_source: Description of training data source (optional)

    Returns:
        ModelRun object
    """
    # Auto-detect run type if not specified
    if run_type is None:
        run_type = "baseline" if use_baseline else "evaluation"

    # Store run type and taxonomy clearly in base_model_id
    base_model_id = f"setfit-{run_type}-{taxonomy}"

    # Store model path or baseline indicator in model_version
    if use_baseline:
        model_version = f"baseline-{taxonomy}"
    else:
        model_version = model_path if model_path else f"unknown-{taxonomy}"

    run = ModelRun(
        started_at=datetime.now().isoformat(),
        status="running",
        run_type=run_type,
        base_model_id=base_model_id,
        model_version=model_version,
        model_type="setfit",
        taxonomy=taxonomy,
        training_data_source=training_data_source,
        hyperparameters_json=json.dumps(config),
    )

    session.add(run)
    session.commit()
    session.refresh(run)

    return run


def main():
    parser = argparse.ArgumentParser(
        description="Run ML classifier predictions on training dataset"
    )
    parser.add_argument(
        "--training_db",
        type=str,
        required=True,
        help="Path to training.db database",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to fine-tuned SetFit model (required unless --use_baseline)",
    )
    parser.add_argument(
        "--use_baseline",
        action="store_true",
        help="Use baseline pre-trained model without fine-tuning",
    )
    parser.add_argument(
        "--taxonomy",
        type=str,
        choices=["v1", "v2", "legacy"],
        default="legacy",
        help="Label taxonomy to use (default: legacy)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test"],
        help="Only evaluate on specific split",
    )
    parser.add_argument(
        "--labeled_only",
        action="store_true",
        help="Only run on samples with labels (for evaluation)",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save predictions to database",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Save predictions to CSV file",
    )
    parser.add_argument(
        "--label_run_id",
        type=int,
        help="Label run ID to use for training labels (defaults to newest)",
    )

    args = parser.parse_args()

    # Validation
    if not args.use_baseline and not args.model_path:
        parser.error("--model_path is required unless --use_baseline is set")

    # Connect to database
    db_path = Path(args.training_db)
    if not db_path.exists():
        raise FileNotFoundError(f"Training database not found: {db_path}")

    engine = create_engine(f"sqlite:///{db_path}")

    # Load samples
    print(f"Loading samples from {db_path}...")
    with Session(engine) as session:
        # Get newest label run if not specified
        effective_label_run_id = args.label_run_id or get_newest_label_run_id(session)
        if effective_label_run_id is None:
            print("Error: No label runs found in database", file=__import__('sys').stderr)
            return
        else:
            print(f"Using specified label run: {effective_label_run_id}")

        samples = load_samples(
            session,
            split=args.split,
            labeled_only=args.labeled_only,
            label_run_id=effective_label_run_id,
        )
        print(f"Loaded {len(samples)} samples")

        if not samples:
            print("No samples found. Exiting.")
            return

        # Initialize classifier
        if args.use_baseline:
            print(f"Initializing baseline SetFit model (taxonomy={args.taxonomy})...")
            classifier = SetFitClassifier(taxonomy=args.taxonomy, use_baseline=True)
        else:
            print(f"Loading fine-tuned model from {args.model_path}...")
            classifier = SetFitClassifier(
                model_path=args.model_path, taxonomy=args.taxonomy
            )

        # Run predictions
        print("Running predictions...")
        predictions, confidences, probabilities = classifier.predict(samples)

        # Get true labels (if available)
        y_true = [s.label for s in samples if s.label]
        y_pred_labeled = [
            pred for i, pred in enumerate(predictions) if samples[i].label
        ]

        # Evaluate
        if y_true and y_pred_labeled:
            print(f"\nEvaluating {len(y_true)} labeled samples...")
            metrics = evaluate_predictions(
                y_true, y_pred_labeled, classifier.labels, verbose=True
            )
        else:
            print("\nNo labeled samples found. Skipping evaluation.")
            metrics = {}

        # Always save metrics to database (even if not saving predictions)
        print("\nSaving metrics to database...")

        # Create training run
        config = {
            "use_baseline": args.use_baseline,
            "model_path": args.model_path,
            "taxonomy": args.taxonomy,
            "split": args.split,
            "labeled_only": args.labeled_only,
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
        run_type_label = "baseline" if args.use_baseline else "fine-tuned"

        run = create_model_run(
            session,
            model_path=args.model_path,
            taxonomy=args.taxonomy,
            use_baseline=args.use_baseline,
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
        run.notes = f"Run type: {run_type_label}, Taxonomy: {args.taxonomy}, Split: {args.split or 'all'}{metrics_summary}"

        session.commit()

        print(
            f"✓ Saved metrics to database (run_id={run.run_id}, type={run_type_label}, taxonomy={args.taxonomy})"
        )
        if metrics:
            print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"  Macro F1: {metrics.get('macro_f1', 0):.4f}")
            print(f"  Weighted F1: {metrics.get('weighted_f1', 0):.4f}")

        # Save predictions to database (optional)
        if args.save_predictions:
            print("\nSaving predictions to database...")

            # Save predictions
            num_saved = save_predictions_to_db(
                session,
                samples,
                predictions,
                confidences,
                probabilities,
                run_id=run.run_id,
                prediction_type=args.split or "all",
            )

            print(f"✓ Saved {num_saved} predictions")

        # Save to CSV
        if args.output_file:
            print(f"\nSaving predictions to {args.output_file}...")
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            import csv

            with output_path.open("w", newline="", encoding="utf-8") as f:
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

            print(f"Saved {len(samples)} predictions to {args.output_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
