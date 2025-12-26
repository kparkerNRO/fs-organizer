# Heuristic Classifier

## Overview

The heuristic classifier is a rule-based classifier that provides initial label predictions for folder classification. It uses patterns from the config files (`variants.yaml`, `creators.yaml`, `filename_cleaning.yaml`) and rules from the classification proposals to assign labels with confidence scores.

## Features

- **Rule-based classification**: Uses predefined patterns and heuristics
- **Confidence scoring**: Each prediction includes a confidence score (0-1)
- **Multi-taxonomy support**: Supports both v1 and v2 taxonomies
- **Fuzzy matching**: Case-insensitive text matching with similarity scoring
- **Context-aware**: Considers depth, parent, children, and sibling context

## Taxonomies

### V1 Taxonomy (Legacy)
- `person_or_group` - Creator or studio
- `content` - Place, category, or content name
- `media_bucket` - Media type or format
- `descriptor` - Variant or modifier
- `other` - Organizational folders
- `unknown` - Cannot be classified

### V2 Taxonomy (Recommended)
- `creator_or_studio` - Creator, publisher, or studio
- `content_subject` - Specific subject matter or location
- `theme_or_genre` - Theme, setting, or genre
- `asset_type` - Type of asset (maps, tokens, etc.)
- `other` - Organizational folders
- `unknown` - Cannot be classified

## Mapping from variants.yaml

The classifier automatically maps variant types from `variants.yaml` to taxonomy labels:

| Variant Type | V1 Label | V2 Label |
|--------------|----------|----------|
| `variant` | `descriptor` | `theme_or_genre` |
| `media_type` | `media_bucket` | `asset_type` |
| `media_format` | `media_bucket` | `asset_type` |

### Examples from variants.yaml

- **Seasons** (type: variant)
  - `winter`, `summer`, `spring`, `fall`, `autumn`
  - → V1: `descriptor`, V2: `theme_or_genre`

- **Media Types** (type: media_type)
  - `VTT`, `Print`, `Music`, `Tiles`, `Animated Scenes`
  - → V1: `media_bucket`, V2: `asset_type`

- **Media Formats** (type: media_format)
  - `PDF`, `Jpeg`, `PNGS`, `WEBP`
  - → V1: `media_bucket`, V2: `asset_type`

- **Layout Variants** (type: variant)
  - `Gridded`, `Gridless`, `No Grid`, `Transparent`, `Interior`, `Exterior`
  - → V1: `descriptor`, V2: `theme_or_genre`

## Classification Rules

### 1. Variant Matching (Confidence: 0.90)
Matches folder names against known variants from `variants.yaml` using fuzzy text matching.

**Example:**
- Folder: "Winter" → `theme_or_genre` (v2)
- Folder: "VTT" → `asset_type` (v2)

### 2. Creator Detection (Confidence: 0.85-0.95)
Detects creators using:
- Known creator names from `creators.yaml`
- Creator keywords (patreon, studios, cartographer, etc.)
- Collaboration markers from `collab_markers`
- Parent context (e.g., "Collaborator Content")
- Depth in hierarchy (higher confidence at shallow depths)

**Example:**
- Folder: "CzePeku" at depth 1 → `creator_or_studio` (0.85)
- Folder: "Collaboration with Tom Cartos" → `creator_or_studio` (0.85)

### 3. Asset Type Detection (Confidence: 0.80-0.90)
Matches common asset type keywords:
- maps, tokens, pack, assets, battlemap, handouts, tiles, music, illustrations

**Example:**
- Folder: "Maps" → `asset_type`
- Folder: "Tokens" with PNG files → `asset_type` (0.90, boosted by file presence)

### 4. Theme Detection (Confidence: 0.75)
Matches theme/genre keywords:
- dungeon, forest, sci-fi, cyberpunk, horror, desert, urban, fantasy, medieval, etc.

**Example:**
- Folder: "Dungeon" → `theme_or_genre`

### 5. Organizational Folders (Confidence: 0.80-0.95)
Detects organizational folders:
- Year folders (2023, 2024, etc.) → 0.95 confidence
- Keywords: rewards, tier, bonus, instructions, guide, notes

**Example:**
- Folder: "2023" → `other` (0.95)
- Folder: "Patreon Rewards" → `other` (0.80)

### 6. Unknown (Confidence: <0.50)
When no rules match, defaults to `unknown` with low confidence.

## Usage

### In Sampling Pipeline

The heuristic classifier is automatically integrated into the sample generation:

```bash
# Generate samples with heuristic predictions (v2 taxonomy, default)
uv run python -m organizer.fine_tuning.cli generate-samples \
    --index-db outputs/run/index.db \
    --output-csv training_samples.csv \
    --sample-size 1000

# Use v1 taxonomy
uv run python -m organizer.fine_tuning.cli generate-samples \
    --index-db outputs/run/index.db \
    --output-csv training_samples.csv \
    --heuristic-taxonomy v1

# Disable heuristic predictions
uv run python -m organizer.fine_tuning.cli generate-samples \
    --index-db outputs/run/index.db \
    --output-csv training_samples.csv \
    --no-heuristic
```

### Programmatic Usage

```python
from fine_tuning.heuristic_classifier import HeuristicClassifier
from utils.config import Config

# Initialize classifier
config = Config()
classifier = HeuristicClassifier(config, taxonomy="v2")

# Classify a single folder
result = classifier.classify(
    name="Winter Maps",
    depth=3,
    parent_name="Maps",
    children_names=["Gridded", "Gridless"],
    file_extensions=["png", "jpg"]
)

print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Reason: {result.reason}")
print(f"Matches: {result.matches}")

# Batch classification
samples = [
    {"name": "Winter", "depth": 3},
    {"name": "CzePeku", "depth": 1},
    {"name": "Maps", "depth": 2}
]
results = classifier.classify_batch(samples)
```

### CSV Output Format

When heuristic predictions are enabled, the CSV includes:

- `heuristic_label` - Predicted label
- `heuristic_confidence` - Confidence score (0.000-1.000)
- `heuristic_reason` - Reasoning for the prediction
- `label` - Empty field for manual labeling

**Example row:**
```csv
name,depth,heuristic_label,heuristic_confidence,heuristic_reason,label
Winter,3,theme_or_genre,0.900,"Matched variant 'winter' from config",
```

## Confidence Score Interpretation

| Confidence | Interpretation | Action |
|------------|----------------|--------|
| 0.90-1.00 | Very High | Likely correct, quick review |
| 0.80-0.89 | High | Generally reliable |
| 0.70-0.79 | Medium | Review carefully |
| 0.50-0.69 | Low | Needs careful review |
| <0.50 | Very Low | Manual labeling required |

## Human Review Process

The heuristic predictions are designed to facilitate human review:

1. **Sort by confidence**: Review low-confidence predictions first
2. **Use as suggestions**: Heuristic labels are pre-filled suggestions
3. **Verify against context**: Check parent, children, siblings for context
4. **Override when needed**: The `label` column is for your final decision

## Extending the Classifier

### Adding New Rules

To add new classification rules, edit `heuristic_classifier.py`:

1. Add pattern sets to the initialization methods (`_build_*_patterns`)
2. Create a new check method (e.g., `_check_new_category`)
3. Add the check to the `classify` method priority chain

### Updating Config Files

The classifier automatically picks up changes to:
- `variants.yaml` - Variant names and types
- `creators.yaml` - Known creator names
- `filename_cleaning.yaml` - Collaboration markers

Just update the YAML files and the classifier will use the new data.

## Testing

Run the test suite:

```bash
cd organizer
uv run pytest fine_tuning/heuristic_classifier_test.py -v
```

The test suite validates:
- Text processing utilities
- Pattern matching logic
- Classification rules for all categories
- Confidence score ranges
- Taxonomy mapping
- Batch processing

## Limitations

1. **Rule-based only**: Cannot learn from data, limited to predefined patterns
2. **No semantic understanding**: Uses text matching, not meaning
3. **Context limitations**: Limited depth of contextual analysis
4. **No contradictions**: Cannot resolve conflicting signals well

## Future Enhancements

Potential improvements:

1. **Semantic similarity**: Use word embeddings for better matching
2. **Probabilistic scoring**: Bayesian combination of multiple signals
3. **Active learning**: Learn from human corrections
4. **Hierarchical rules**: Better use of parent-child relationships
5. **Domain adaptation**: Customize rules per creator or collection

## Related Files

- `heuristic_classifier.py` - Main classifier implementation
- `heuristic_classifier_test.py` - Test suite
- `training_utils.py` - Integration with sampling pipeline
- `cli.py` - Command-line interface
- `config/variants.yaml` - Variant definitions
- `config/creators.yaml` - Creator names
- `config/filename_cleaning.yaml` - Cleaning rules and collaboration markers
