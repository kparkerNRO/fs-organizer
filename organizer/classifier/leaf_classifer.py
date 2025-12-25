# train_setfit_leaf_typing_hardnegs.py
import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# Hard-negative (batch-hard) loss
from sentence_transformers.losses import BatchHardSoftMarginTripletLoss  # mines hardest negs in-batch


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

# If you only want to mine confusers for these (often most ambiguous), keep this list.
DEFAULT_CONFUSABLE = {"primary_author", "secondary_author", "collection", "subject"}

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
    """
    Returns:
      - model text with explicit parents/leaf separation
      - leaf_key: normalized leaf tokens (for confuser mining)
    """
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


def read_csv_rows(path: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV must have headers: path,label")
        for r in reader:
            p = (r.get("path") or "").strip()
            y = (r.get("label") or "").strip()
            if p and y:
                rows.append((p, y))
    return rows


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
    """
    Oversample confusable examples by duplicating:
      - an anchor example
      - its top-k most leaf-similar examples from *different labels*
    This increases the chance that mined "hard negatives" appear together in batches.
    """
    # Precompute trigram sets
    tri = [char_trigrams(lk) for lk in train_leaf_keys]

    # Bucket indices by label for faster scanning
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

        # Find best negatives by leaf similarity across other labels
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

        # Duplicate anchor + negatives to boost co-occurrence
        for _ in range(factor):
            extra_texts.append(train_texts[i])
            extra_labels.append(y)
            for j in picked:
                extra_texts.append(train_texts[j])
                extra_labels.append(train_labels[j])

    return extra_texts, extra_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with headers path,label")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--output_dir", default="./leaf_classifier_setfit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)

    # Training knobs
    ap.add_argument("--batch_size", type=int, default=32)  # good default for triplet batches
    ap.add_argument("--num_epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)

    # Triplet (hard-negative) mining knobs
    ap.add_argument("--use_triplet_loss", action="store_true", default=True)
    ap.add_argument("--samples_per_label", type=int, default=2)

    # Oversampling mined confusers knobs
    ap.add_argument("--hardneg_k", type=int, default=2)
    ap.add_argument("--hardneg_min_sim", type=float, default=0.25)
    ap.add_argument("--hardneg_factor", type=int, default=2)
    ap.add_argument(
        "--hardneg_labels",
        default="primary_author,secondary_author,collection,subject",
        help="Comma-separated label names to mine/oversample",
    )

    args = ap.parse_args()

    # Basic validation: batch_size multiple of samples_per_label for triplet batching
    if args.use_triplet_loss and (args.batch_size % args.samples_per_label != 0):
        raise ValueError("When using triplet loss, batch_size must be a multiple of samples_per_label.")

    label2id: Dict[str, int] = {l: i for i, l in enumerate(LABELS)}
    id2label: Dict[int, str] = {i: l for l, i in label2id.items()}

    rows = read_csv_rows(args.data)
    if not rows:
        raise ValueError("No training rows found.")

    unknown = {y for _, y in rows if y not in label2id}
    if unknown:
        raise ValueError(f"Unknown labels found: {sorted(unknown)}")

    texts: List[str] = []
    leaf_keys: List[str] = []
    labels: List[int] = []
    for p, y in rows:
        t, lk = path_to_leaf_example(p)
        texts.append(t)
        leaf_keys.append(lk)
        labels.append(label2id[y])

    # Split
    idx = list(range(len(labels)))
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=labels
    )

    train_texts = [texts[i] for i in train_idx]
    train_leaf_keys = [leaf_keys[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    test_ds = Dataset.from_dict(
        {"text": [texts[i] for i in test_idx], "label": [labels[i] for i in test_idx]}
    )

    # Ensure triplet loss constraints: need >=2 examples per class in training
    if args.use_triplet_loss:
        counts = defaultdict(int)
        for y in train_labels:
            counts[y] += 1
        too_small = [id2label[y] for y, c in counts.items() if c < 2]
        if too_small:
            raise ValueError(
                "Triplet loss requires >=2 train examples per label. "
                f"These labels have <2 in the train split: {too_small}"
            )

    # Hard-negative oversampling (train split only)
    confusable = {s.strip() for s in args.hardneg_labels.split(",") if s.strip()}
    extra_texts, extra_labels = augment_with_hard_negatives(
        train_texts=train_texts,
        train_leaf_keys=train_leaf_keys,
        train_labels=train_labels,
        id2label=id2label,
        confusable_labels=confusable,
        k=args.hardneg_k,
        min_sim=args.hardneg_min_sim,
        factor=args.hardneg_factor,
    )

    if extra_texts:
        train_texts = train_texts + extra_texts
        train_labels = train_labels + extra_labels

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})

    # Model + trainer
    model = SetFitModel.from_pretrained(args.model)

    trainer_kwargs = dict(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        column_mapping={"text": "text", "label": "label"},
    )

    # Triplet loss = true batch-hard negative mining in the contrastive phase
    if args.use_triplet_loss:
        trainer_kwargs.update(
            loss=BatchHardSoftMarginTripletLoss,
            samples_per_label=args.samples_per_label,
        )

    trainer = SetFitTrainer(**trainer_kwargs)
    trainer.train()

    # Evaluate
    y_true = test_ds["label"]
    y_pred = trainer.model.predict(test_ds["text"])
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\nMacro-F1: {macro_f1:.4f}\n")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[id2label[i] for i in range(len(LABELS))],
            digits=4,
        )
    )

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    print(f"\nSaved to: {args.output_dir}")
    if extra_texts:
        print(f"Hard-negative oversampling added {len(extra_texts)} extra training rows.")


if __name__ == "__main__":
    main()
