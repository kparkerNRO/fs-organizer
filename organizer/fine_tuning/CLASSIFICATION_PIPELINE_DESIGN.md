# Classification Pipeline Design for Hobby-Scale Projects

## 1. Introduction & Problem Statement

This document outlines the design for a classification pipeline aimed at automatically organizing a personal collection of RPG assets (maps, tokens, etc.). The primary challenge is to create a robust classification system that is effective even with a relatively small, hobby-scale dataset.

The core problem is that digital asset collections from various creators lack a standardized folder structure. A folder's name might be a creator's name ("Tom Cartos"), a content theme ("Cyberpunk"), or a specific location ("The Sunken City"). This pipeline is designed to assign a single, meaningful category to each folder, creating a structured library from a chaotic one.

## 2. Core Design Principles

Our design is guided by the constraints of a small, evolving dataset and the need for high accuracy without massive manual labeling efforts.

- **Structural and Contextual Features over Name Matching**: We avoid relying solely on folder names. Instead, the model will prioritize structural features (e.g., path depth, parent/child folder names, file types within a folder). This is crucial for distinguishing between a folder named after a creator (e.g., `/Tom Cartos/`) and a folder containing their content (e.g., `/Tom Cartos/Maps/The Forbidden Temple/`).

- **Two-Stage Approach**: We use a hybrid model to maximize accuracy and minimize labeling effort:
    1.  **Heuristic Baseline**: A rule-based classifier provides an initial, high-quality prediction. It codifies domain-specific knowledge and serves as a strong starting point.
    2.  **ML Fine-Tuning**: A machine learning model is then trained on manually-verified data to learn the more nuanced patterns that the heuristic model misses.

- **V2 Taxonomy as Output Schema**: The pipeline classifies folders into one of six clear, distinct categories defined in the "V2 Taxonomy." This schema is designed to be more intuitive and useful for organizing RPG assets. The classes are:
    - `creator_or_studio`
    - `content_subject`
    - `descriptor`
    - `asset_type`
    - `other`
    - `unknown`

## 3. Baseline Model: The Heuristic Classifier

The first stage is a rule-based heuristic classifier that provides initial predictions with confidence scores. This model is transparent, easy to debug, and provides a solid performance baseline (expected 40-60% Macro F1).

- **Rule-Based Classification**: The classifier uses a series of prioritized rules to assign a label. For example, rules for organizational folders run before creator detection to correctly label a folder like "Patreon Rewards" as `other` instead of `creator_or_studio`.

- **Classification Rules with Examples**:
    - **Variant Matching**: Matches folder names against a predefined list of variants.
        - *Source*: `variants.yaml`
        - *Example*: A folder named "Winter" or "Gridless" is mapped to `descriptor`. A folder named "VTT" is mapped to `asset_type`.
        - *Confidence*: High (0.90)

    - **Creator Detection**: Identifies creators using keywords, known names, and folder structure.
        - *Source*: `creators.yaml`, keywords like `patreon`, `studios`.
        - *Example*: A folder named "CzePeku" at a shallow depth is classified as `creator_or_studio`.
        - *Confidence*: High (0.85-0.95), especially at low directory depths.

    - **Asset Type Detection**: Matches common asset type keywords.
        - *Keywords*: `maps`, `tokens`, `pack`, `assets`, `battlemap`, `handouts`.
        - *Example*: A folder named "Tokens" containing PNG files is classified as `asset_type`.
        - *Confidence*: Medium-High (0.80-0.90).

    - **Theme Detection**: Matches common genre or setting keywords.
        - *Keywords*: `dungeon`, `forest`, `sci-fi`, `cyberpunk`, `horror`, `desert`.
        - *Example*: A folder named "Dungeon" is classified as `descriptor`.
        - *Confidence*: Medium (0.75).

    - **Organizational Folders**: Detects administrative or meta-folders.
        - *Keywords*: `rewards`, `tier`, `bonus`, `instructions`, or year patterns (e.g., "2023").
        - *Example*: A folder named "2024" or "Patreon Rewards" is classified as `other`.
        - *Confidence*: High (0.80-0.95).

- **Confidence Scoring**: Each prediction is assigned a confidence score. This allows us to prioritize manual review, focusing on low-confidence predictions first and accelerating the labeling process.

## 4. ML Model for Fine-Tuning (Small-Data Focus)

The second stage uses a machine learning model to learn from the shortcomings of the heuristic classifier. This stage is designed specifically for few-shot learning, making it ideal for a hobby project.

- **Feature Engineering**: The primary input to the model is a synthesized text string derived from the folder's context, including its name, parent's name, and the names of its children. This encodes the structural information vital for accurate classification.

- **Why SetFit for Few-Shot Learning**: We use the `SetFit` framework because it is exceptionally effective on small datasets. It works by fine-tuning a powerful Sentence Transformer model with a contrastive learning objective, allowing it to achieve high accuracy with very few labeled examples per class. This dramatically reduces the burden of manual data labeling.

- **PRACTICAL Techniques for Small Datasets**:
    - **Handling Class Imbalance**: Our dataset is naturally imbalanced (e.g., many `content_subject` folders, few `creator_or_studio` folders). To prevent the model from ignoring rare classes, we will use **class weights** in the loss function, which penalizes misclassifications of minority classes more heavily.
    - **Preventing Overfitting**: With a small dataset, overfitting is a major risk. We will use a dedicated **validation set** to monitor performance and employ **early stopping**. This halts the training process when the model's performance on the validation set stops improving, ensuring we capture the model at its peak generalization performance.
    - **Efficient Labeling**: To make the best use of limited labeling time, we can employ techniques like **active learning**. By caching the **prediction entropy** of the model, we can identify which unlabeled folders the model is most "confused" about. These high-entropy samples are the most valuable ones to label next.

- **Sample Size Recommendations**: For a project of this scale, an initial labeled dataset of **600-1200 samples** is a realistic and effective target to achieve strong performance with the SetFit model.

## 5. Metrics for Success

Choosing the right metric is critical for understanding the model's true performance.

- **Why Macro F1-Score over Accuracy**: Accuracy is a misleading metric for imbalanced datasets. A model could achieve high accuracy by simply always predicting the most common class. The **Macro F1-Score** is our primary metric because it calculates the F1-score (a balance of precision and recall) for each class independently and then averages them. This provides a much more robust measure of the model's ability to perform well across all classes, including the rare ones.

- **Target Performance**:
    - **Baseline (Heuristic) Model**: Expected Macro F1 of **40-60%**.
    - **Final (ML) Model**: Target Macro F1 of **82-85%**.
