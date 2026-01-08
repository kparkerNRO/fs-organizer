# Design Doc: Enhanced Folder Name Grouping Algorithm

**Date:** 2026-01-07

**Status:** Proposed

## 1. Overview

This document outlines a design to update the folder name grouping algorithm in `organizer/stages/grouping/folder_name_grouping.py`. The goal is to replace the current rigid, common-prefix-based grouping with a more flexible, hierarchical decomposition system.

The new algorithm will:
- Process each folder name independently to find its optimal constituent parts.
- Use a confidence score to evaluate and choose between multiple potential ways to split a name.
- Implement a set of heuristic rules for scoring, such as penalizing stopword-only groups and splits that result in a small prefix followed by a large suffix.
- Achieve folder grouping as an emergent property: similar folder names will be decomposed into similar sets of parts.

## 2. High-Level Plan

The implementation will be centered around three main changes:

1.  **Refactor `apply_folder_name_grouping`**: The main function will be simplified to orchestrate the new decomposition process. It will iterate through each name from the previous stage, apply a new decomposition function, and store the results.
2.  **Create `decompose_name` function**: This new recursive function will be the core of the logic. It will take a single name and find the best way to hierarchically split it into multiple parts by repeatedly finding the highest-confidence binary split.
3.  **Create `calculate_split_confidence` function**: This helper will score potential splits of a name based on the specified heuristic rules, enabling the `decompose_name` function to choose the best option.

## 3. Detailed Component Design

### A. `apply_folder_name_grouping` (Updated)

This function will be modified to drive the new decomposition process for each `GroupCategoryEntry`.

```python
# In organizer/stages/grouping/folder_name_grouping.py

def apply_folder_name_grouping(
    session: Session,
    run_id: int,
    snapshot_id: int,
) -> None:
    """
    Apply hierarchical name decomposition to break down folder names into meaningful parts.
    This replaces the previous common prefix grouping with a more flexible, rule-based
    decomposition that is applied to each name individually.
    """
    logger.info("Applying hierarchical folder name grouping")

    # (Setup of new iteration remains the same)
    # ...

    # Get entries from the previous iteration
    previous_entries = session.scalars(select(GroupCategoryEntry).where(...)).all()
    if not previous_entries:
        return

    for entry in previous_entries:
        processed_name = entry.processed_name or entry.pre_processed_name
        if not processed_name:
            continue

        # Decompose the name into its constituent parts
        decomposed_parts = decompose_name(processed_name)

        # Create new entries for each decomposed part
        for part in decomposed_parts:
            if not part or not part.strip():
                continue

            new_entry = GroupCategoryEntry(
                folder_id=entry.folder_id,
                iteration_id=iteration_id,
                pre_processed_name=entry.pre_processed_name,
                processed_name=part,
                path=entry.path,
                confidence=entry.confidence, # Confidence can be refined here
                processed=False,
                derived_names=entry.derived_names,
            )
            session.add(new_entry)

    session.commit()
    logger.info("Hierarchical folder name grouping complete")
```

### B. `decompose_name` (New Function)

This function will recursively split a name into its highest-confidence parts.

```python
# In organizer/stages/grouping/folder_name_grouping.py

from typing import List, Tuple

# A confidence threshold to decide whether to accept a split
DECOMPOSITION_CONFIDENCE_THRESHOLD = 0.6

def decompose_name(name: str) -> List[str]:
    """
    Recursively decomposes a name into its best constituent parts based on a confidence score.
    Returns a list of strings representing the decomposed parts.
    """
    words = name.split()
    if len(words) <= 1:
        return [name]

    best_split = None
    max_confidence = -1.0

    # Find the best binary split for the current name
    for i in range(1, len(words)):
        part1 = " ".join(words[:i])
        part2 = " ".join(words[i:])

        confidence = calculate_split_confidence(part1, part2, name)

        if confidence > max_confidence:
            max_confidence = confidence
            best_split = (part1, part2)

    # If the best split is good enough, recurse on the second part
    if max_confidence > DECOMPOSITION_CONFIDENCE_THRESHOLD and best_split:
        # Return the first part, plus the result of decomposing the second part
        return [best_split[0]] + decompose_name(best_split[1])
    else:
        # If no split is confident enough, do not decompose the name further
        return [name]
```

### C. `calculate_split_confidence` (New Function)

This function will contain the scoring logic based on the provided rules, acting as the "brain" for the decomposition.

```python
# In organizer/stages/grouping/folder_name_grouping.py
# Assumes STOP_WORDS constant is available

def calculate_split_confidence(part1: str, part2: str, original_name: str) -> float:
    """
    Calculates a confidence score for a proposed binary split of a name.
    The score is based on a set of heuristic rules.
    """
    score = 1.0

    # Rule: First part should generally be longer than the second
    if len(part1) <= len(part2):
        # Apply a penalty, stronger if the first part is much shorter
        score *= 0.8 * (len(part1) / (len(part2) + 1e-6))

    # Rule: The last section should not be more than 50% of the original length
    if len(part2) > 0.5 * len(original_name):
        score *= 0.7

    # Rule: Penalize groups that are mostly stopwords
    part1_words = [w for w in part1.lower().split() if w not in STOP_WORDS]
    part2_words = [w for w in part2.lower().split() if w not in STOP_WORDS]

    # The prefix (part1) must contain non-stopwords
    if not part1_words:
        return 0.0  # A prefix of only stopwords is invalid

    # Penalize if the second part has more meaningful content than the first
    if len(" ".join(part1_words)) < len(" ".join(part2_words)):
        score *= 0.9

    return score
```
