# Fine-Tuning Module

This module provides ML classifier training and prediction capabilities for the fs-organizer project.

## CLI Commands

All fine-tuning commands are available through the main `organizer.py` CLI under the `model` subcommand:

```bash
# Show available model commands
uv run python organizer.py model --help
```

### 1. Extract Features

Extract classification features from all nodes in a snapshot and populate training.db:

```bash
uv run python organizer.py model extract-features \
  --index-db outputs/run/index.db \
  --training-db outputs/training.db \
  --snapshot-id 1
```

**Parameters:**
- `--index-db`: Path to index.db containing the snapshot
- `--training-db`: Path to training.db (will be created if doesn't exist)
- `--snapshot-id`: Which snapshot to extract features from
- `--batch-size`: Number of samples to insert per batch (default: 1000)

**Output:** training.db populated with TrainingSample records containing feature vectors for all folders

---

### 2. Generate Training Samples

Generate a CSV of diverse training samples for manual labeling:

```bash
uv run python organizer.py model generate-samples \
  --index-db outputs/run/index.db \
  --output-csv training_samples.csv \
  --snapshot-id 1 \
  --sample-size 1000 \
  --diversity-factor 0.7
```

**Parameters:**
- `--index-db`: Path to index.db containing the snapshot
- `--output-csv`: Where to save the CSV for manual labeling
- `--snapshot-id`: Which snapshot to sample from
- `--sample-size`: Number of samples to generate (default: 800)
- `--min-depth`: Minimum folder depth (default: 1)
- `--max-depth`: Maximum folder depth (default: 10)
- `--diversity-factor`: 0-1, higher = more diverse (default: 0.7)

**Output:** CSV file with columns for manual labeling including context (parent, siblings, children, file extensions)

---

### 3. Train Model

Train a SetFit classifier on labeled data:

```bash
uv run python organizer.py model train \
  --data training_samples_labeled.csv \
  --output-dir ./models/classifier_v1 \
  --num-epochs 8 \
  --batch-size 32
```

**Parameters:**
- `--data`: Path to CSV with labeled samples (must have `path` and `label` columns)
- `--output-dir`: Where to save the trained model (default: `./leaf_classifier_setfit`)
- `--model`: Base sentence transformer model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `--batch-size`: Training batch size (default: 32)
- `--num-epochs`: Number of epochs (default: 6)
- `--learning-rate`: Learning rate (default: 2e-5)
- `--samples-per-label`: Samples per label for triplet loss (default: 2)
- `--test-size`: Fraction for test set (default: 0.2)
- `--hardneg-k`: Number of hard negatives to mine (default: 2)
- `--hardneg-min-sim`: Similarity threshold for hard negatives (default: 0.25)
- `--hardneg-factor`: Oversampling factor (default: 2)
- `--hardneg-labels`: Labels to mine hard negatives for (default: primary_author,secondary_author,collection,subject)
- `--no-triplet-loss`: Disable triplet loss

**Output:**
- Trained SetFit model saved to output directory
- Evaluation metrics (Macro F1, classification report, confusion matrix)

---

### 4. Run Predictions

Run a trained model against a dataset and evaluate:

```bash
uv run python organizer.py model predict \
  --training-db outputs/training.db \
  --model-path ./models/classifier_v1 \
  --split test \
  --labeled-only \
  --save-predictions
```

**Parameters:**
- `--training-db`: Path to training.db with samples
- `--model-path`: Path to trained SetFit model (required unless --use-baseline)
- `--use-baseline`: Use baseline pre-trained model without fine-tuning
- `--taxonomy`: Label taxonomy: v1, v2, or legacy (default: legacy)
- `--split`: Only run on specific split: train, validation, or test
- `--labeled-only`: Only run on samples with labels (for evaluation)
- `--save-predictions`: Save predictions to database
- `--output-file`: Export predictions to CSV file

**Output:**
- Evaluation metrics (if labeled samples available)
- Predictions saved to database (if `--save-predictions`)
- CSV export (if `--output-file` specified)

---

## Label Taxonomies

The module supports three label taxonomies:

### Legacy (default)
Used by existing training code:
- `primary_author` - Main creator/publisher
- `secondary_author` - Collaborator/guest creator
- `collection` - Campaign, product line, or themed collection
- `subject` - Specific map/asset subject matter
- `media_format` - Technical format (VTT, Print, etc.)
- `media_type` - Media category (Maps, Tokens, Music, etc.)
- `variant` - Variations (Gridded, Night, Clean, etc.)
- `other` - Administrative/organizational folders

### V1 (from classification_proposal.md)
- `person_or_group` - Creator/publisher
- `content` - Place/category/themed content
- `media_bucket` - Media type container
- `descriptor` - Variant/modifier
- `other` - Organizational
- `unknown` - Ambiguous

### V2 (from classification_proposal_v2.md)
- `creator_or_studio` - Creator/publisher entity
- `content_subject` - Specific subject matter
- `theme_or_genre` - Setting/genre/theme
- `asset_type` - Type of asset
- `other` - Organizational
- `unknown` - Ambiguous

---

## Complete Workflows

### Workflow A: Baseline Evaluation (No Fine-Tuning)

Get baseline performance metrics using a pre-trained model before investing in labeling and fine-tuning:

```bash
# 1. Gather files into snapshot
uv run python organizer.py gather /path/to/patreon/assets --storage outputs/run

# 2. Extract features to training.db
uv run python organizer.py model extract-features \
  --index-db outputs/run/index.db \
  --training-db outputs/training.db \
  --snapshot-id 1

# 3. Generate a small labeled sample set for baseline evaluation
uv run python organizer.py model generate-samples \
  --index-db outputs/run/index.db \
  --output-csv baseline_eval.csv \
  --snapshot-id 1 \
  --sample-size 200

# 4. Manually label the CSV (fill in the 'label' column)
# Edit baseline_eval.csv - only 200 samples for quick baseline

# 5. Import labels to training.db
uv run python organizer.py training apply-classifications \
  baseline_eval.csv \
  --storage outputs/run

# 6. Run baseline model evaluation (no fine-tuning)
uv run python organizer.py model predict \
  --training-db outputs/training.db \
  --use-baseline \
  --taxonomy legacy \
  --labeled-only \
  --output-file baseline_results.csv

# Review baseline_results.csv to see baseline performance
# If baseline is good enough, you may not need fine-tuning!
# If baseline is poor, proceed with full fine-tuning workflow
```

**Why run baseline first?**
- See if pre-trained model is already good enough
- Understand which classes are difficult
- Decide if fine-tuning investment is worthwhile
- Get a performance target to beat

---

### Workflow B: Full Fine-Tuning Pipeline

Complete workflow from data gathering to fine-tuned model:

```bash
# 1. Gather files into snapshot
uv run python organizer.py gather /path/to/patreon/assets --storage outputs/run

# 2. Extract features to training.db
uv run python organizer.py model extract-features \
  --index-db outputs/run/index.db \
  --training-db outputs/training.db \
  --snapshot-id 1

# 3. Generate training samples for labeling
uv run python organizer.py model generate-samples \
  --index-db outputs/run/index.db \
  --output-csv training_samples.csv \
  --snapshot-id 1 \
  --sample-size 1000

# 4. Manually label the CSV (fill in the 'label' column)
# Edit training_samples.csv and add labels to each row
# This is the most time-consuming step - aim for 800-1200 labels

# 5. Train a fine-tuned model
uv run python organizer.py model train \
  --data training_samples.csv \
  --output-dir ./models/my_classifier \
  --num-epochs 8 \
  --batch-size 32

# 6. Evaluate fine-tuned model on test set
# (The train command automatically creates a test split and evaluates)

# 7. Apply fine-tuned model to all unlabeled data
uv run python organizer.py model predict \
  --training-db outputs/training.db \
  --model-path ./models/my_classifier \
  --save-predictions \
  --output-file predictions.csv

# 8. Review predictions and identify low-confidence samples
# Filter predictions.csv for low confidence scores
# Add these to your training set and retrain (active learning)
```

---

### Workflow C: Baseline vs Fine-Tuned Comparison

Compare baseline and fine-tuned model performance:

```bash
# 1-4. Same as Workflow B (gather, extract, generate, label)

# 5. Split your labeled data into train/test manually
# Or use the CSV to create separate files: train.csv and test.csv

# 6. Import test labels to training.db with split='test'
uv run python organizer.py training apply-classifications \
  test.csv \
  --storage outputs/run

# 7. Evaluate baseline model
uv run python organizer.py model predict \
  --training-db outputs/training.db \
  --use-baseline \
  --taxonomy legacy \
  --split test \
  --labeled-only \
  --output-file baseline_test_results.csv

# Note the Macro F1 score from baseline

# 8. Train fine-tuned model (on train.csv)
uv run python organizer.py model train \
  --data train.csv \
  --output-dir ./models/my_classifier \
  --num-epochs 8

# 9. Evaluate fine-tuned model
uv run python organizer.py model predict \
  --training-db outputs/training.db \
  --model-path ./models/my_classifier \
  --split test \
  --labeled-only \
  --output-file finetuned_test_results.csv

# Compare Macro F1 scores between baseline and fine-tuned
# Fine-tuned should outperform baseline significantly
```

---

## Module Structure

```
fine_tuning/
├── cli.py                    # Main typer CLI (integrated into organizer.py)
├── __main__.py               # Allow running as module
├── run_classifier.py         # Prediction and evaluation logic
├── leaf_classifer.py         # Legacy training script (converted to cli.py)
├── feature_extraction.py     # Feature extraction from nodes
├── training_utils.py         # Training data utilities
├── README.md                 # This file
└── MIGRATION.md              # Migration guide from argparse to typer
```

---

## Notes

- **Always run baseline evaluation first** - It's quick and tells you if fine-tuning is needed
- Baseline model uses `sentence-transformers/all-MiniLM-L6-v2` without task-specific training
- Typical baseline Macro F1: 40-60% (depends on data)
- Target fine-tuned Macro F1: 82-85% (per proposals)
- Predictions are tracked in the `training.db` database
- Models use SetFit (few-shot learning with sentence transformers)
- Hard negative mining improves performance on confusable classes
- Feature extraction is required before any predictions can be made
