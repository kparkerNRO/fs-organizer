# [UPDATED] Proposal: 6-Class Folder Classifier (Version 2)

This document outlines a revised plan to build a classifier for categorizing folders within an RPG asset library. It incorporates feedback from practical and technical reviews to create a more robust, realistic, and effective model.

## 1. The Problem

Organizing thousands of RPG assets from various creators is a manual, time-consuming process. A multi-class classifier can automate this by assigning a semantic type to each folder, transforming a chaotic file system into a structured database. This proposal details the plan to build and validate such a classifier.

## 2. [UPDATED] Class Definitions

This section has been significantly revised to resolve contradictions identified in the practical review, providing clearer, more distinct, and more useful categories.

- **`creator_or_studio`**: **(Formerly `person_or_group`)** This class is now strictly for identifying the creator, publisher, or studio. It refers to the entity that *produces* the content, not the content itself. This resolves the "Collaborator Content" confusion.
  - **Example**: `Abyssal Brews`, `Tom Cartos`, `Cze and Peku`
  - **Critical Distinction**: A folder named `Abyssal Brews` (the creator) is classed `creator_or_studio`, while a folder for a map they created, `Abyssal Falls`, is `content_subject`. The model must learn this from structural cues.

- **`content_subject`**: **(Formerly `content`)** This class identifies the specific subject matter or named location of an asset. It answers the question, "What is this a map/token of?"
  - **Example**: `Abyssal Falls`, `The Sunken City`, `Forest Camp`

- **`descriptor`**: **(Formerly `descriptor`)** This class captures the setting, genre, or general theme of an asset. It's broader than `content_subject` and provides useful filtering context.
  - **Example**: `Dungeon`, `Cyberpunk`, `Horror`, `Desert`

- **`asset_type`**: **(Formerly `media_bucket`)** This class describes the *type* of asset in the folder, clarifying its intended use. This is more intuitive than "media bucket" and less ambiguous than using file extensions.
  - **Example**: `Maps`, `Tokens`, `Assets`, `Handouts`, `Pack`

- **`other`**: This class is for folders that have a clear purpose but don't fit into the other categories. This often includes administrative, meta, or date-based folders.
  - **Example**: `Patreon Rewards`, `2023`, `Instructions`

- **`unknown`**: This class is for folders whose purpose cannot be determined from their name, structure, or contents.

## 3. Feature Engineering: Structural Not Nominal

We will maintain the original approach of using structural features (path depth, file counts, child folder names, etc.) instead of relying on simple name matching. This is critical for teaching the model the difference between a `creator_or_studio` and a `content_subject`, as both may share keywords (e.g., "Abyssal"). The model will learn that a creator's name often appears at a higher level in the directory structure, with content folders nested underneath.

## 4. [UPDATED] Baseline Model

The original baseline was too simplistic. The new baseline will be a rule-based model with more sophisticated, RPG-specific heuristics to provide a more competitive benchmark.

- **`creator_or_studio` Rules**:
  - If folder name contains `patreon`, `studios`, `co` -> `creator_or_studio`
  - If folder name matches a known creator list -> `creator_or_studio`
- **`asset_type` Rules**:
  - If folder name contains `maps`, `tokens`, `pack`, `assets`, `battlemap` -> `asset_type`
- **`descriptor` Rules**:
  - If folder name contains `dungeon`, `forest`, `sci-fi`, `cyberpunk`, `horror` -> `descriptor`
- **`other` Rules**:
  - If folder name is a year (e.g., `2023`, `2024`) -> `other`
  - If folder name contains `rewards`, `tier` -> `other`


### [NEW] Handling Class Imbalance

Our dataset is likely to be imbalanced, with many `content_subject` folders and few `creator_or_studio` folders. To prevent the model from ignoring smaller classes, we will use class weights in the loss function.

```python
# [NEW] Example: Calculating and using class weights in PyTorch
import torch
from torch import nn

# Example counts for each of our 6 classes
# (creator, content, theme, asset_type, other, unknown)
class_counts = torch.tensor([50, 150, 70, 80, 40, 10], dtype=torch.float)
total_samples = class_counts.sum()

# Calculate weights as the inverse of class frequency
class_weights = total_samples / class_counts
weights_tensor = class_weights.to(device) # device is 'cuda' or 'cpu'

# Use the weights in the loss function
# This will penalize errors on under-represented classes more heavily
loss_function = nn.CrossEntropyLoss(weight=weights_tensor)
```

### [NEW] Entropy Caching for Active Learning

To focus manual labeling efforts, we can cache the model's prediction entropy for each folder. High entropy indicates low confidence, flagging folders that would be most valuable to label next.

```python
# [NEW] Example: Simple entropy caching for active learning
import torch.nn.functional as F

entropy_cache = {} # { "folder_path": 0.95, ... }
NEEDS_REVIEW_THRESHOLD = 0.9 # Tune this threshold

def calculate_entropy(logits):
    """Calculates the entropy of a single prediction."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.item()

# During classification runs:
# for path, features in dataset:
#     if path not in entropy_cache:
#         logits = model(features)
#         entropy = calculate_entropy(logits)
#         entropy_cache[path] = entropy
#
#     if entropy_cache[path] > NEEDS_REVIEW_THRESHOLD:
#         print(f"'{path}' needs manual review (entropy: {entropy_cache[path]:.2f})")
```

### [NEW] Validation and Early Stopping

To prevent overfitting on our small dataset, we will use a dedicated validation set (~15% of labeled data). We will monitor the validation loss after each epoch and stop training if the loss fails to improve for a set number of epochs (e.g., a patience of 3-5). This ensures we save the model at its point of peak performance.

## 6. [UPDATED] Metrics and Success Criteria

The metrics have been updated to reflect the challenge of class imbalance and to set a more realistic goal.

- **Primary Metric**: **Macro F1-Score**. Accuracy is a poor metric for imbalanced datasets, as a model can achieve high accuracy by simply predicting the majority class. Macro F1-Score calculates the F1 score for each class independently and averages them, providing a much better measure of the model's performance across all classes, including rare ones.
- **Success Target**: A **Macro F1-Score of 82-85%**. This is a more realistic and challenging goal than the original 87% accuracy target, reflecting the complexity of the task and the imbalanced nature of the data.
- **Secondary Metric**: Per-class F1 scores will be monitored to identify specific classes where the model is struggling.

## 7. Out of Scope

- **No change**: This project will not perform detailed creator name matching or build a knowledge base of creators. The classification will remain based on the structural features of the file system.
