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
  --index-db data/index.db \
  --training-db data/training.db
```
This creates a new database `outputs/training.db` populated with feature vectors for all your folders.

#### Step 3: Generate Evaluation Samples (30 sec)

Generate a small, diverse set of samples for manual labeling. 200 is enough for a quick and statistically relevant evaluation.

```bash
uv run python organizer.py model generate-samples \
  --index-db data/index.db \
  --output-csv outputs/baseline_eval.csv \
  --sample-size 200
```

#### Step 4: Label Samples (10-20 min)

Open `baseline_eval.csv` in a spreadsheet editor. Your job is to fill in the `label` column for each row based on the folder's name and context provided in the other columns. Use the labels defined in the **Label Taxonomy** section below.

Once you've labeled the samples, import them back into the training database:

```bash
uv run python organizer.py model apply-classifications outputs/baseline_eval.csv --storage data
```

#### Step 5: Evaluate Baseline Model (1 min)

Now, run the pre-trained baseline model against your labeled samples to see how well it performs.

```bash
uv run python organizer.py model predict \
  --training-db data/training.db \
  --use-baseline \
  --labeled-only \
  --output-file baseline_results.csv
```

The console will output a detailed performance report, including the key **Macro F1-Score**.

---

## Full Fine-Tuning Workflow

If your baseline evaluation shows room for improvement (typically a Macro F1 score below 70%), it's time to fine-tune a model on your own data. This teaches the model the specific patterns and nuances of your collection.

### Step 1: Generate a Larger Training Set

Generate a larger set of samples for labeling. 800-1200 samples is a good target for a high-performing model.

```bash
uv run python organizer.py model generate-samples \
  --index-db data/index.db \
  --output-csv training_samples.csv \
  --snapshot-id 1 \
  --sample-size 1000
```

### Step 2: Label Your Data

This is the most time-consuming part. Open `training_samples.csv` and, just like in the baseline step, fill in the `label` column for all the samples.

### Step 3: Train the Fine-Tuned Model

Once your data is labeled, point the training command at your CSV file. This process can take 5-15 minutes.

```bash
uv run python organizer.py model train \
  --data training_samples.csv \
  --output-dir ./models/my_classifier \
  --num-epochs 8
```
The command will automatically split your data, train the model, run an evaluation, and save the final model to the specified directory.

### Step 4: Apply Your New Model

With a trained model, you can now run it on all the unlabeled folders in your database.

```bash
uv run python organizer.py model predict \
  --training-db outputs/training.db \
  --model-path ./models/my_classifier \
  --save-predictions \
  --output-file all_predictions.csv
```
This will generate a `predictions.csv` file containing the predicted label for every folder.

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

### `extract-features`
Extracts classification features from a snapshot and populates the training database.

**Usage:**
```bash
uv run python organizer.py model extract-features \
    --index-db <path/to/index.db> \
    --training-db <path/to/training.db> \
    --snapshot-id <id>
```
**Key Parameters:**
*   `--index-db`: Path to the source `index.db` created by the `gather` command.
*   `--training-db`: Path to the target `training.db` where features will be stored.
*   `--snapshot-id`: (Optional) The snapshot to process. Defaults to the latest one.

### `generate-samples`
Generates a CSV of diverse folder samples for manual labeling.

**Usage:**
```bash
uv run python organizer.py model generate-samples \
    --index-db <path/to/index.db> \
    --output-csv <path/to/samples.csv> \
    --sample-size 1000
```
**Key Parameters:**
*   `--output-csv`: The path where the generated CSV file will be saved.
*   `--sample-size`: The number of samples to generate (e.g., 200 for baseline, 1000+ for fine-tuning).
*   `--diversity-factor`: (0-1) A factor to control the diversity of samples. Higher is more diverse.

### `train`
Trains a SetFit classifier on a labeled CSV file.

**Usage:**
```bash
uv run python organizer.py model train \
    --data <path/to/labeled_samples.csv> \
    --output-dir <path/to/save/model> \
    --num-epochs 8
```
**Key Parameters:**
*   `--data`: Path to your CSV file containing `path` and `label` columns.
*   `--output-dir`: Directory where the trained model will be saved.
*   `--num-epochs`: Number of training epochs (default: 6). More epochs can lead to better performance but also risk overfitting.
*   `--test-size`: Fraction of data to hold out for testing (default: 0.2).

### `predict`
Runs a trained model (or the baseline) against a dataset to generate predictions.

**Usage:**
```bash
# Using a fine-tuned model
uv run python organizer.py model predict \
    --training-db <path/to/training.db> \
    --model-path <path/to/your/model>

# Using the baseline model for evaluation
uv run python organizer.py model predict \
    --training-db <path/to/training.db> \
    --use-baseline --labeled-only
```
**Key Parameters:**
*   `--model-path`: Path to a fine-tuned model directory.
*   `--use-baseline`: If set, uses the pre-trained baseline model instead of a fine-tuned one.
*   `--labeled-only`: Use this for evaluation. It runs predictions only on samples that have a ground-truth label.
*   `--save-predictions`: If set, saves the output predictions back into the database.
*   `--output-file`: (Optional) Exports all predictions to a CSV file.

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