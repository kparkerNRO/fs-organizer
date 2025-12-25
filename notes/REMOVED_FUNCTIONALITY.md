# Historical Record: Removed Functionality and ML Approaches

**Document Purpose**: This document records functionality, ML techniques, and algorithmic approaches that were implemented, tested, and subsequently removed from the fs-organizer codebase. This serves as institutional knowledge for future development.

**Date**: December 2025
**Cleanup Branch**: `claude/cleanup-unused-code-QhTbf`

---

## Summary

During a comprehensive code cleanup, **~876 lines** of legacy code were removed across **11 commits**. This included deprecated algorithms, alternative ML approaches, and superseded functionality. The removed code represents earlier experiments and implementations that were replaced by more effective solutions.

---

## 1. Legacy String-Based Grouping Algorithm (`legacy_group.py`)

**Removed in**: `a63c048` (348 lines)
**Status**: Completely unused, replaced by NLP-based approach

### Functionality

A deterministic, rule-based algorithm for grouping similar folder names using string manipulation and pattern matching.

### Key Algorithms

#### 1.1 Token-Based Name Grouping
- **Function**: `group_similar_names(names_to_group: list)`
- **Approach**: Greedy token matching with common prefix detection
- **Algorithm**:
  1. Split folder names into space-separated tokens
  2. Find common prefix tokens between names
  3. Create hierarchical groupings based on shared prefixes
  4. Handle overlapping token sets by merging to shortest common token

**Example**:
```
"Castle Tower Day" + "Castle Tower Night" → Castle Tower/Day, Castle Tower/Night
"Snow Inn" + "Snow Tavern" → Snow/Inn, Snow/Tavern
```

#### 1.2 View Type Splitting
- **Function**: `split_view_type(base_name)`
- **Approach**: Pattern matching against predefined view types
- **Used Constants**: `VIEW_TYPES` (never defined - breaking dependency)
- **Purpose**: Separate semantic content from view metadata (gridded/gridless, day/night)

#### 1.3 Filename Cleaning
- **Function**: `clean_filename(base_name, creator_removes, ...)`
- **Approach**: Multi-stage regex-based cleaning
- **Operations**:
  - Remove creator-specific prefixes (e.g., "$5 Rewards", "Admiral")
  - Normalize underscores to spaces
  - Remove "Part N" and "Pt. N" numbering
  - Strip file dimensions (e.g., "1920x1080")
  - Remove leading/trailing numbers and special characters

#### 1.4 Recursive Folder Grouping
- **Function**: `group_nested_folder(file_to_str_to_group: dict)`
- **Approach**: Recursive application of token grouping
- **Algorithm**: Apply `group_similar_names` then recursively process sub-groups

### Why It Was Removed

1. **Brittle**: Relied on hardcoded constants (`VIEW_TYPES`, `GROUPING_EXCEPTIONS`) that were never properly defined
2. **Language-Specific**: Assumed English space-separated tokens
3. **Deterministic**: No ML/statistical component, couldn't handle spelling variations
4. **Replaced By**: NLP-based clustering with embedding similarity (in `nlp_grouping.py`)

### What Replaced It

- **NLP-based clustering** using sentence embeddings
- **Distance metrics** combining text similarity + structural path information
- **Statistical approaches** instead of hardcoded rules

---

## 2. File-Based Configuration System

**Removed in**: `44ea8bb` (100 lines)
**File**: `utils/config.py`

### Functionality

A TOML-based configuration system for controlling organizer behavior.

### Implementation

```python
class Config:
    def __init__(self, config_file=None, **kwargs):
        # Organization behavior
        self.zip_backup_state: ZipBackupState
        self.file_backup_state: FileBackupState
        self.preserve_modules: bool

        # Execution behavior
        self.unzip: bool
        self.organize: bool
        self.should_execute: bool

        # File handling
        self.creators_to_exceptions: dict
        self.keywords_to_remove: dict
        self.folder_names_to_rename: dict

        # File structure
        self.input_dir: Path
        self.output_dir: Path
        self.levels_to_preserve: int
```

### Features

- **TOML parsing**: `_parse_config_file()` loaded settings from `.toml` files
- **CLI override**: `_parse_inputs()` allowed command-line args to override config
- **Enum-based states**: Used `ZipBackupState` and `FileBackupState` enums
- **Multi-source config**: Combined file-based + CLI argument configuration

### Why It Was Removed

1. **Never Used**: No code called the Config class
2. **Dependency Issues**: Referenced undefined `FileBackupState` and `ZipBackupState` enums
3. **Superseded**: Configuration moved to dictionary-based constants in same file

### What Replaced It

Simple module-level dictionaries:
- `CREATOR_REMOVES` - Creator-specific cleanup rules
- `GROUPING_EXCEPTIONS` - Terms to exclude from grouping
- `KNOWN_VARIANT_TOKENS` - Known variant descriptors

---

## 3. Heuristic Name Categorization

**Removed in**: `f52a629` (112 lines)
**File**: `pipeline/classify.py`

### Functionality

Two-stage heuristic algorithm for classifying folder names into categories and variants.

### Algorithms

#### 3.1 Name Parsing
- **Function**: `parse_name(name: str) -> tuple[list[str], list[str]]`
- **Approach**: Split by hyphens, classify tokens as categories or variants
- **Algorithm**:
  1. Clean filename
  2. Split on hyphens into components
  3. For each component:
     - Check if token matches `KNOWN_VARIANT_TOKENS`
     - If yes → variant; else → category
  4. Return separate lists of categories and variants

#### 3.2 Heuristic Categorization
- **Function**: `heuristic_categorize(session) -> None`
- **Approach**: Database-wide classification with hierarchical inference
- **Algorithm**:
  1. Parse all folder names into categories/variants
  2. Assign classification:
     - Only variants → `ClassificationType.VARIANT`
     - One category, no variants → `ClassificationType.CATEGORY`
     - Mixed → `ClassificationType.UNKNOWN`
  3. Create `PartialNameCategory` entries for each token
  4. **Hierarchical inference**: If folder has one category AND all children are variants → promote to `ClassificationType.SUBJECT`

### Machine Learning Aspect

**None** - This was a purely rule-based heuristic, not ML-based.

### Why It Was Removed

1. **Hardcoded Rules**: Relied on pre-defined variant token list
2. **No Learning**: Couldn't adapt to new data patterns
3. **Limited Accuracy**: Hyphen-based splitting was too simplistic
4. **Never Called**: Function existed but wasn't invoked in production code

### What Replaced It

- **`classify_folders()`**: Frequency-based classification
- **Statistical Analysis**: Uses folder name frequency across dataset
- **Parent-Child Analysis**: Considers hierarchical relationships
- **No Hardcoded Tokens**: Learns patterns from data distribution

---

## 4. Folder-Scoped Grouping Algorithm

**Removed in**: `68523b8` (60 lines)
**File**: `grouping/group.py`

### Functionality

A specialized grouping algorithm that only clustered items within the same parent folder.

### Algorithm

#### 4.1 Folder-Within Grouping
- **Function**: `group_within_folder(session, text_distance_ratio=0.6)`
- **Approach**: Constrained clustering with parent folder boundary
- **ML Technique**: DBSCAN-style clustering with custom distance metric
- **Algorithm**:
  1. Get all entries from previous iteration
  2. Cluster using `cluster_with_custom_metric()`
  3. **Key constraint**: Use `compute_same_folder_distance_matrix()`
  4. Refine groups using standard refinement

#### 4.2 Same-Folder Distance Matrix
- **Function**: `compute_same_folder_distance_matrix(folders, text_distance_ratio)`
- **ML Technique**: Custom distance metric for constrained clustering
- **Algorithm**:
  ```python
  for i, j in all_pairs:
      if parent(i) == parent(j):
          # Standard cosine distance on text embeddings
          distance = 1.0 - cosine_similarity(text_vec_i, text_vec_j)
      else:
          # Infinite distance - prevents cross-folder clustering
          distance = 1e6  # Effectively infinite
  ```

### Why It Was Removed

1. **Too Restrictive**: Parent folder constraint was overly limiting
2. **Commented Out**: Disabled in production code (line 397 of `group.py`)
3. **Only Tested**: Used in `test_group_iteration` but nowhere else
4. **Better Alternative**: Global clustering with path distance worked better

### What It Tried to Solve

**Problem**: Prevent unrelated items in different folders from being grouped together.

**Example**:
```
/maps/castle/tower  }
/maps/castle/keep   } ← Should group (same folder)
/monsters/tower     } ← Should NOT group (different context)
```

### Why It Failed

The hard boundary prevented legitimate groupings where similar content appeared in different organizational structures. The solution was too rigid.

### What Replaced It

- **`compute_custom_distance_matrix()`**: Incorporates path distance as weighted factor
- **Soft Penalty**: Path distance adds to similarity metric rather than hard boundary
- **Global Optimization**: Allows cross-folder grouping when semantically justified

---

## 5. Iteration-Based Name Grouping

**Removed in**: `a8b44f4` (108 lines)
**File**: `grouping/group.py`

### Functionality

A single-iteration grouping algorithm using text similarity clustering.

### Algorithm

- **Function**: `group_by_name(session, text_distance_ratio)`
- **ML Technique**: Hierarchical clustering with custom distance threshold
- **Approach**: One-pass clustering based purely on name similarity
- **Algorithm**:
  1. Fetch entries from iteration N-1
  2. Prepare records (extract text embeddings)
  3. Compute custom distance matrix:
     ```python
     distance = α × text_distance + (1-α) × (struct_distance / (1 + struct_distance))
     ```
  4. Apply clustering
  5. Refine groups
  6. Commit as iteration N

### Text Distance Ratio

**Parameter**: `text_distance_ratio` (typically 0.9)
- **Purpose**: Weight text similarity vs. structural similarity
- **High values (0.9)**: Prioritize name similarity over folder structure
- **Low values (0.6)**: Give more weight to folder hierarchy

### Why It Was Removed

1. **Commented Out**: Disabled at line 404 of `group.py`
2. **Test-Only**: Only used in `test_group_iteration`
3. **Superseded**: Production uses multi-stage pipeline instead

### What Replaced It

A **multi-stage pipeline** in `group_folders()`:
1. `process_folders_to_groups()` - Initial processing
2. `pre_process_groups()` - Hyphen-splitting preprocessing
3. `decompose_compound_tags()` - **NEW**: N-gram tag decomposition
4. `compact_groups()` - Final consolidation

The single-iteration approach was replaced by a more sophisticated multi-stage pipeline with specialized processing at each step.

---

## 6. Class-Based Tag Decomposition API

**Removed in**: `95c2c06` (68 lines)
**File**: `grouping/tag_decomposition_test.py`

### Functionality

An object-oriented API for tag decomposition using a `TagDecomposer` class.

### Proposed Interface

```python
class TagDecomposer:
    def __init__(self, min_confidence=0.3):
        self.min_confidence = min_confidence
        self.component_stats = None
        self.tag_vectors = None

    def extract_component_statistics(self, entries):
        """Extract word frequencies and contexts"""
        pass

    def compute_tag_vectors(self, entries):
        """Compute embeddings for each tag"""
        pass

    def generate_decomposition_candidates(self, entries):
        """Find compound tags and propose decompositions"""
        pass
```

### Test Coverage (Removed)

1. **`test_component_statistics_extraction`**: Tested word frequency analysis
2. **`test_decomposition_candidate_generation`**: Tested compound tag detection

### Why It Was Never Implemented

1. **Design Changed**: Switched to functional API instead of OOP
2. **Tests Written First**: TDD approach where tests preceded implementation
3. **Better Alternative Found**: Simpler functional interface was sufficient

### What Replaced It

**Functional API** (in `tag_decomposition.py`):
- `decompose_compound_tags(session)` - Main entry point
- `extract_component_statistics()` - Standalone function
- `generate_decomposition_candidates()` - Standalone function

The functional approach avoided unnecessary state management and made the code simpler.

---

## 7. Empty Data Model Placeholder

**Removed in**: `6d8fd91` (8 lines)
**File**: `data_models/categorize.py`

### Commented Code

```python
# @dataclass
# class Group():
#     id: int
#     name: str
#     item_count: int
#     categories: dict[str, Category]
#     folders: list[Folder]
```

### Purpose

A skeleton for a potential `Group` data model, never implemented.

### Why It Was Removed

1. **Never Implemented**: Only a commented skeleton
2. **Not Imported**: No code referenced it
3. **Unclear Purpose**: Overlapped with existing `GroupCategory` model

---

## Key Machine Learning Techniques That Were Tried and Removed

### 1. **Deterministic Token Matching** (legacy_group.py)
- **Approach**: Greedy common-prefix algorithm
- **Result**: ❌ Too brittle, language-specific
- **Replaced with**: Embedding-based similarity

### 2. **Hard Boundary Clustering** (group_within_folder)
- **Approach**: Infinite distance for cross-folder pairs
- **Result**: ❌ Too restrictive, prevented valid groupings
- **Replaced with**: Soft distance penalties

### 3. **Heuristic Token Classification** (heuristic_categorize)
- **Approach**: Rule-based category vs. variant detection
- **Result**: ❌ Required maintaining hardcoded token lists
- **Replaced with**: Statistical frequency analysis

### 4. **Single-Iteration Clustering** (group_by_name)
- **Approach**: One-pass hierarchical clustering
- **Result**: ❌ Insufficient for complex data
- **Replaced with**: Multi-stage pipeline with preprocessing

### 5. **Class-Based Decomposition API** (TagDecomposer)
- **Approach**: OOP design with stateful object
- **Result**: ❌ Unnecessary complexity
- **Replaced with**: Simple functional interface

---

## Current Production Algorithms (for comparison)

### Active NLP/ML Techniques

1. **Sentence Transformers**: Embedding-based text similarity
2. **Custom Distance Metrics**: Weighted combination of text + structural distance
3. **DBSCAN-style Clustering**: `cluster_with_custom_metric()` with adaptive thresholds
4. **N-gram Analysis**: Tag decomposition using component frequency
5. **Statistical Classification**: Frequency-based folder classification
6. **Hierarchical Refinement**: Multi-pass group consolidation

### Active Pipeline (group_folders)

```
Input Folders
    ↓
process_folders_to_groups()  ← Initial embedding-based clustering
    ↓
pre_process_groups()         ← Hyphen splitting
    ↓
decompose_compound_tags()    ← N-gram decomposition (NEW)
    ↓
compact_groups()             ← Final consolidation
    ↓
Output Groups
```

---

## Lessons Learned

### What Worked

✅ **Embedding-based similarity** over token matching
✅ **Soft distance penalties** over hard boundaries
✅ **Multi-stage pipelines** over single-pass algorithms
✅ **Functional APIs** over object-oriented for stateless operations
✅ **Statistical learning** over hardcoded rules

### What Didn't Work

❌ Deterministic rule-based grouping
❌ Hardcoded token lists (VIEW_TYPES, GROUPING_EXCEPTIONS)
❌ Parent-folder hard boundaries
❌ File-based configuration with never-defined enums
❌ Class-based APIs for simple stateless operations

---

## References

For implementation details of current algorithms:
- **NLP Grouping**: `organizer/grouping/nlp_grouping.py`
- **Tag Decomposition**: `organizer/grouping/tag_decomposition.py`
- **Classification**: `organizer/pipeline/classify.py`
- **Main Pipeline**: `organizer/grouping/group.py`

---

**Document Version**: 1.0
**Last Updated**: December 25, 2025
**Cleanup Commits**: 11 commits, ~876 lines removed
