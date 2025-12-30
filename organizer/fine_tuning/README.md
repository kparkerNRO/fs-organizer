# Fine-Tuning Module

This module provides a complete workflow for training, evaluating, and using machine learning classifiers to automatically categorize folders in your filesystem.

The primary goal is to take a chaotic directory structure and assign a meaningful label (e.g., `creator_or_studio`, `asset_type`, `content_subject`) to each folder.

## Baseline Evaluation


#### Step 1: Gather Data (2 min)

First, create a snapshot of your filesystem. This scans your target directory and stores its structure in a database.

```bash
uv run python organizer.py gather /path/to/your/assets --storage data
```

This creates a database at `data/index.db`.

#### Step 2: Extract Features (1-5 min)

Next, process the snapshot to extract features (like folder depth, parent names, file extensions) that the model will use for classification.

```bash
uv run python organizer.py model extract-features \
  --snapshot-id 1
```
This reads `data/index.db` and creates/updates `data/training.db` with feature vectors for all folders in the snapshot.

#### Step 3: Generate Evaluation Samples (30 sec)

Generate a small, diverse set of samples for manual labeling. 200 is enough for a quick and statistically relevant evaluation.

```bash
uv run python organizer.py model generate-samples \
  --snapshot-id 1 \
  --output-csv outputs/baseline_eval.csv \
  --config organizer/config/fine_tuning/generate_samples.json
```
The sample size and diversity settings come from the JSON config file.

#### Step 4: Label Samples (10-20 min)

Open `baseline_eval.csv` in a spreadsheet editor. Your job is to fill in the `label` column for each row based on the folder's name and context provided in the other columns. Use the labels defined in the **Label Taxonomy** section below.

Once you've labeled the samples, import them back into the training database:

```bash
uv run python organizer.py model apply-classifications \
  --input-csv outputs/baseline_eval.csv \
  --config organizer/config/fine_tuning/apply_classifications.json
```

#### Step 5: Evaluate Baseline Model (1 min)

Now, run the pre-trained baseline model against your labeled samples to see how well it performs.

```bash
uv run python organizer.py model predict \
  --config organizer/config/fine_tuning/predict_baseline.json
```
In your config file, set `use_baseline=true` and `labeled_only=true` to evaluate only labeled samples.

The console will output a detailed performance report, including the key **Macro F1-Score**.

---

## Full Fine-Tuning Workflow

If your baseline evaluation shows room for improvement (typically a Macro F1 score below 70%), it's time to fine-tune a model on your own data. This teaches the model the specific patterns and nuances of your collection.

### Step 1: Generate a Larger Training Set

Generate a larger set of samples for labeling. 800-1200 samples is a good target for a high-performing model.

```bash
uv run python organizer.py model generate-samples \
  --snapshot-id 1 \
  --output-csv training_samples.csv \
  --config organizer/config/fine_tuning/generate_samples.json
```

### Step 2: Label Your Data

This is the most time-consuming part. Open `training_samples.csv` and, just like in the baseline step, fill in the `label` column for all the samples.

### Step 3: Train the Fine-Tuned Model

Once your data is labeled, train your model using the training database. This process can take 5-15 minutes.

```bash
uv run python organizer.py model train \
  --config organizer/config/fine_tuning/train.json \
  --model-path ./models/my_classifier
```

If you have multiple label runs, you can specify which one to use:

```bash
uv run python organizer.py model train \
  --label-run-id 3 \
  --config organizer/config/fine_tuning/train.json \
  --model-path ./models/my_classifier
```

The command will automatically:
- Load labeled samples from the database
- Split your data into train/test sets
- Train the model with hard negative mining
- Run evaluation and save the final model to the specified directory

### Step 4: Apply Your New Model

With a trained model, you can now run it on all the unlabeled folders in your database.

```bash
uv run python organizer.py model predict \
  --model-path ./models/my_classifier \
  --config organizer/config/fine_tuning/predict.json
```
This writes predictions into `data/training.db` and logs evaluation metrics if labels exist.

---

## Label Taxonomy

This is the canonical set of labels used for classification.

*   `creator_or_studio`: The entity (person or company) that produces the content.
    *   *Example*: `Tom Cartos`, `Cze and Peku`, `Abyssal Brews`

*   `content_subject`: The specific subject matter of the asset. Answers "What is this a map/token of?"
    *   *Example*: `The Sunken City`, `Forest Camp`, `Dragon's Lair`

*   `descriptor`: A broader category describing the setting, genre, or general theme.
    *   *Example*: `Dungeon`, `Cyberpunk`, `Horror`, `Desert`

*   `asset_type`: The type of asset and its intended use.
    *   *Example*: `Maps`, `Tokens`, `Assets`, `Handouts`, `Pack`

*   `other`: Administrative, meta, or organizational folders that don't fit other categories.
    *   *Example*: `Patreon Rewards`, `2023`, `Instructions`

*   `unknown`: The folder's purpose cannot be determined from its name or context.

**Heuristic Predictions:** When you generate sample CSVs, they may include a `heuristic_prediction` column. This is a "best guess" from a simple rule-based system, which can help speed up your manual labeling process. The fine-tuned model's predictions will be much more accurate and will also come with a `confidence_score`.

---

## CLI Command Reference

All commands are run via `uv run python organizer.py model <COMMAND>`.

Each command defaults to a config file under `organizer/config/fine_tuning/`. Pass `--config` to override the default file.

### Base Path Override (Debug Only)

The fine-tuning CLI uses `StorageSettings.base_path` to resolve default data and model folders. You normally should not change this, but you can override it for debugging or special runs by setting the environment variable on the command line:

```bash
FS_ORGANIZER_BASE_PATH=/tmp/fs-organizer \
  uv run python organizer.py model train
```

### `extract-features`
Extracts classification features from a snapshot and populates the training database.

**Usage:**
```bash
uv run python organizer.py model extract-features \
    --snapshot-id <id> \
    --config organizer/config/fine_tuning/extract_features.json
```
**Key Parameters:**
*   `--snapshot-id`: (Optional) The snapshot to process. Defaults to the latest one.
*   `--config`: (Optional) JSON config for feature extraction (batch sizes, caps). Defaults are used if omitted.

### `generate-samples`
Generates a CSV of diverse folder samples for manual labeling.

**Usage:**
```bash
uv run python organizer.py model generate-samples \
    --snapshot-id <id> \
    --output-csv <path/to/samples.csv> \
    --config organizer/config/fine_tuning/generate_samples.json
```
**Key Parameters:**
*   `--snapshot-id`: (Optional) Snapshot to sample from. Defaults to the latest one.
*   `--output-csv`: The path where the generated CSV file will be saved.
*   `--config`: (Optional) JSON config for sample size, depth range, and diversity.

### `train`
Trains a SetFit classifier on labeled samples stored in the training database.

**Usage:**
```bash
uv run python organizer.py model train \
    --config organizer/config/fine_tuning/train.json \
    --model-path <path/to/save/model> \
    --label-run-id <id>
```
**Key Parameters:**
*   `--config`: (Optional) JSON config for hyperparameters (epochs, batch size, hard negatives).
*   `--model-path`: (Optional) Override the default model directory.
*   `--label-run-id`: (Optional) Label run to use (defaults to newest).

### `predict`
Runs a trained model (or the baseline) against samples in the training database.

**Usage:**
```bash
# Using a fine-tuned model
uv run python organizer.py model predict \
    --model-path <path/to/your/model> \
    --config organizer/config/fine_tuning/predict.json

# Using the baseline model for evaluation
uv run python organizer.py model predict \
    --config organizer/config/fine_tuning/predict_baseline.json
```
**Key Parameters:**
*   `--config`: (Optional) JSON config with `use_baseline` and `labeled_only`.
*   `--model-path`: Path to a fine-tuned model directory (required unless `use_baseline=true`).
*   `--label-run-id`: (Optional) Label run to evaluate against (defaults to newest).
*   `--split`: (Optional) Only run on a specific split: `train`, `validation`, or `test`.

Predictions and evaluation results are stored in `data/training.db`.

### `zero-shot`
Runs zero-shot classification using the taxonomy without fine-tuning.

**Usage:**
```bash
uv run python organizer.py model zero-shot \
    --config organizer/config/fine_tuning/zero_shot.json \
    --label-run-id <id> \
    --split <train|validation|test>
```
**Key Parameters:**
*   `--config`: (Optional) JSON config with `labeled_only`.
*   `--label-run-id`: (Optional) Label run to evaluate against (defaults to newest).
*   `--split`: (Optional) Only run on a specific split: `train`, `validation`, or `test`.

### `apply-classifications`
Validates and imports a labeled samples CSV into the training database.

**Usage:**
```bash
uv run python organizer.py model apply-classifications \
    --input-csv <path/to/labeled_samples.csv> \
    --config organizer/config/fine_tuning/apply_classifications.json
```
**Key Parameters:**
*   `--input-csv`: Path to the labeled samples CSV (defaults to `<storage_path>/samples.csv`).
*   `--config`: (Optional) JSON config for `labeler`, `split`, and `validate_only`.

---

## Interpreting Results

After running an evaluation, you'll get a report in your console. Here’s how to interpret it.

### Macro F1-Score
This is your most important metric. It measures the model's average performance across all classes, giving equal weight to each. This is much more reliable than accuracy, especially when some classes are rare.

### Decision Guide

| Macro F1 Score | Decision                                    | Expected Improvement        |
|----------------|---------------------------------------------|-----------------------------|
| > 70%          | **Fine-tuning may not be needed.**          | -                           |
| 60-70%         | **Fine-tuning optional.**                   | Moderate gain (~10-20%)     |
| 40-60%         | **Fine-tuning recommended.**                | Significant gain (~20-30%)  |
| < 40%          | **Fine-tuning strongly recommended.**       | High potential for improvement |

A typical fine-tuned model should achieve a Macro F1 score of **82-85%** or higher.

### Confusion Matrix
The report also includes a list of the most common errors, like this:
```
MOST COMMON ERRORS
  collection           -> subject              : 8 errors
  subject              -> collection           : 6 errors
```
This tells you that the model is most frequently confusing `collection` and `subject`. This is incredibly valuable feedback, indicating that you should focus on providing clearer examples of these two classes during labeling.
