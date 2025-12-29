from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from .common import load_samples
from .taxonomy import get_labels
from utils.text_processing import char_trigrams, jaccard_similarity


@dataclass
class TrainingConfig:
    """Configuration for training a SetFit model."""
    taxonomy: str
    output_dir: Path
    model_id: str
    num_epochs: int
    label_run_id: int
    test_size: float
    seed: int
    hardneg_k: int
    hardneg_min_sim: float
    hardneg_factor: int
    hardneg_labels: str


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


def train_model(session: Session, config: TrainingConfig):
    """Train a SetFit classifier."""
    LABELS = sorted(get_labels(config.taxonomy))

    # Load labeled samples only
    samples = load_samples(
        session, split=None, labeled_only=True, label_run_id=config.label_run_id
    )

    if not samples:
        raise ValueError("No labeled training samples found in database")

    print(f"Loaded {len(samples)} labeled training samples")

    # Validate labels
    label2id: Dict[str, int] = {l: i for i, l in enumerate(LABELS)}
    id2label: Dict[int, str] = {i: l for l, i in label2id.items()}

    unknown = {s.label for s in samples if s.label and s.label not in label2id}
    if unknown:
        raise ValueError(f"Unknown labels found: {sorted(unknown)}")

    # Extract features from samples
    print("Preparing training data...")
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
    print(f"Splitting data (test_size={config.test_size})...")
    idx = list(range(len(labels)))
    train_idx, test_idx = train_test_split(
        idx, test_size=config.test_size, random_state=config.seed, stratify=labels
    )

    train_texts = [texts[i] for i in train_idx]
    train_leaf_keys = [leaf_keys[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    test_ds = Dataset.from_dict(
        {"text": [texts[i] for i in test_idx], "label": [labels[i] for i in test_idx]}
    )

    # Hard-negative oversampling
    if not config.hardneg_labels.strip():
        if config.taxonomy == "legacy":
            default_labels = ["primary_author", "secondary_author", "collection", "subject"]
        elif config.taxonomy == "v1":
            default_labels = ["person_or_group", "content"]
        else:  # v2
            default_labels = ["creator_or_studio", "content_subject"]
        confusable = set(default_labels)
    else:
        confusable = {s.strip() for s in config.hardneg_labels.split(",") if s.strip()}

    print(f"Mining hard negatives for labels: {', '.join(confusable)}...")

    extra_texts, extra_labels = _augment_with_hard_negatives(
        train_texts=train_texts,
        train_leaf_keys=train_leaf_keys,
        train_labels=train_labels,
        id2label=id2label,
        confusable_labels=confusable,
        k=config.hardneg_k,
        min_sim=config.hardneg_min_sim,
        factor=config.hardneg_factor,
    )

    if extra_texts:
        print(f"Added {len(extra_texts)} hard negative samples")
        train_texts = train_texts + extra_texts
        train_labels = train_labels + extra_labels

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})

    # Train model
    print(f"Initializing model: {config.model_id}")
    setfit_model = SetFitModel.from_pretrained(config.model_id)

    trainer_kwargs = dict(
        model=setfit_model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        num_epochs=config.num_epochs,
        column_mapping={"text": "text", "label": "label"},
    )

    trainer = SetFitTrainer(**trainer_kwargs)

    print(f"\nTraining for {config.num_epochs} epochs...")
    trainer.train()

    # Evaluate
    print("\nEvaluating on test set...")
    y_true = test_ds["label"]
    y_pred = trainer.model.predict(test_ds["text"])
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n{'=' * 80}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"{'=' * 80}\n")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[id2label[i] for i in range(len(LABELS))],
            digits=4,
        )
    )

    # Save model
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(config.output_dir))
    print(f"\n✓ Model saved to: {config.output_dir}")

    if extra_texts:
        print(f"✓ Hard-negative oversampling added {len(extra_texts)} samples")