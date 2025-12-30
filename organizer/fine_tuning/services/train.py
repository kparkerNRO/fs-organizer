import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset
from fine_tuning.settings import CommonSettings
from fine_tuning.taxonomy import get_labels
from fine_tuning.services.common import get_effective_label_run_id, load_samples
from pydantic import BaseModel, Field
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from storage.manager import StorageManager
from utils.text_processing import char_trigrams, jaccard_similarity

logger = logging.getLogger(__name__)


class TrainSettings(BaseModel):
    """Settings for the 'train' command."""

    output_dir: Path = Field(
        default=Path("./leaf_classifier_setfit"),
        description="Directory to save trained model",
    )
    model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Base sentence transformer model",
    )
    batch_size: int = Field(
        32,
        description="Batch size for training (must be multiple of samples_per_label)",
    )
    num_epochs: int = Field(
        4,
        description="Number of training epochs",
    )
    learning_rate: float = Field(
        2e-5,
        description="Learning rate",
    )
    samples_per_label: int = Field(
        2,
        description="Samples per label for triplet loss batching",
    )
    hardneg_k: int = Field(
        2,
        description="Number of hard negatives to mine per anchor",
    )
    hardneg_min_sim: float = Field(
        0.25,
        description="Minimum similarity threshold for hard negative mining",
    )
    hardneg_factor: int = Field(
        2,
        description="Oversampling factor for hard negatives",
    )
    hardneg_labels: str = Field(
        "",
        description="Comma-separated labels to mine hard negatives for (defaults to all confusable labels in taxonomy)",
    )
    test_size: float = Field(
        0.2,
        description="Fraction of data to use for testing (0.0-1.0)",
    )
    seed: int = Field(
        42,
        description="Random seed for reproducibility",
    )
    no_triplet_loss: bool = Field(
        False,
        description="Disable triplet loss (use default SetFit loss)",
    )
    no_hard_negatives: bool = Field(
        False,
        description="Disable hard negative mining and oversampling",
    )


class FullTrainingSettings(CommonSettings, TrainSettings):
    pass


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


def prepare_training_data(
    settings: FullTrainingSettings, manager: StorageManager
) -> Tuple[Dataset, Dataset, Dict[int, str]]:
    """Loads, processes, and splits data into training and test datasets."""
    try:
        labels_list = sorted(get_labels(settings.taxonomy))
    except ValueError as e:
        logger.error(f"Invalid taxonomy '{settings.taxonomy}': {e}")
        raise

    label2id = {label: i for i, label in enumerate(labels_list)}
    id2label = {i: label for label, i in label2id.items()}

    logger.info(f"Loading training data from {manager.get_training_db_path()}...")
    with manager.get_training_session() as session:
        effective_label_run_id = get_effective_label_run_id(
            session, settings.label_run_id
        )
        samples = load_samples(
            session, labeled_only=True, label_run_id=effective_label_run_id
        )

    if not samples:
        logger.error("No labeled training samples found in database")
        raise ValueError("No labeled training samples found in database")
    logger.info(f"Loaded {len(samples)} labeled training samples")

    unknown_labels = {s.label for s in samples if s.label and s.label not in label2id}
    if unknown_labels:
        logger.error(f"Unknown labels found: {sorted(unknown_labels)}")
        logger.error(f"Valid labels: {', '.join(labels_list)}")
        raise ValueError(f"Unknown labels found: {sorted(unknown_labels)}")

    texts = [s.text for s in samples if s.label]
    leaf_keys = [s.name_norm for s in samples if s.label]
    labels = [label2id[s.label] for s in samples if s.label]

    logger.info(f"Splitting data (test_size={settings.test_size})...")
    indices = list(range(len(labels)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=settings.test_size,
        random_state=settings.seed,
        stratify=labels,
    )

    train_texts = [texts[i] for i in train_idx]
    train_leaf_keys = [leaf_keys[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    test_ds = Dataset.from_dict(
        {"text": [texts[i] for i in test_idx], "label": [labels[i] for i in test_idx]}
    )

    if not settings.no_hard_negatives:
        confusable = {
            s.strip() for s in settings.hardneg_labels.split(",") if s.strip()
        }
        logger.info(f"Mining hard negatives for labels: {', '.join(confusable)}...")
        extra_texts, extra_labels = augment_with_hard_negatives(
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
            logger.info(f"Added {len(extra_texts)} hard negative samples")
            train_texts += extra_texts
            train_labels += extra_labels

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    return train_ds, test_ds, id2label


def train_model(settings: FullTrainingSettings, manager: StorageManager) -> None:
    train_ds, test_ds, id2label = prepare_training_data(settings, manager)

    logger.info(f"Initializing model: {settings.model}")
    model = SetFitModel.from_pretrained(settings.model)

    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        num_epochs=settings.num_epochs,
        column_mapping={"text": "text", "label": "label"},
    )
    logger.info(f"Training for {settings.num_epochs} epochs...")
    trainer.train()

    logger.info("Evaluating on test set...")
    y_true = test_ds["label"]
    y_pred = trainer.model.predict(test_ds["text"])
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    logger.info(f"{'=' * 80}")
    logger.info(f"Macro F1-Score: {macro_f1:.4f}")
    logger.info(f"{'=' * 80}")
    logger.info(
        classification_report(
            y_true,
            y_pred,
            target_names=[id2label[i] for i in range(len(id2label))],
            digits=4,
        )
    )

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(settings.output_dir))
    logger.info(f"Model saved to: {settings.output_dir}")
