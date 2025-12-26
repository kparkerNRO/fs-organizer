# Folder Classification System Redesign Proposal

## Executive Summary

This proposal outlines a comprehensive redesign of the file organization classification system to classify folders into six semantic categories: person_or_group (creator), content, media_bucket, descriptor, other, and unknown. The analysis is based on 3,512 folders from Patreon RPG asset collections.

## 1. Label Taxonomy Refinement

### Proposed Label Definitions

Based on analysis of the sample data, here are clear definitions for each category:

#### 1. `person_or_group` (Creator)
**Definition**: Folders representing creators, authors, or publishing entities - including both primary creators and collaborators.
- **Examples**:
  - Primary: "Tom Cartos", "Borough Bound", "CzePeku", "Limithron"
  - Nested/Collab: "Limithron - The Juggernaut", "Collaboration with Cze & Peku", "Tabletop RPG Music - The Engines of Aether"
- **Characteristics**:
  - Primary creators typically at depth 1-2, contain diverse portfolios
  - Collaborators may appear deeper (depth 3-4) within "Collaborator Content" or similar folders
  - Often contains person/studio names, collaboration markers
- **Decision Rules** (pattern-based, NOT name-matching):
  - Contains highly diverse content types (multiple media formats, varied themes)
  - Sibling folders often represent other creators (not variants of same content)
  - Has collab markers ("Collaboration with", "-", "by", "ft", "&") OR
  - High folder-to-file ratio with thematic diversity across children OR
  - Appears in dedicated collaboration/attribution context (parent named "Collaborator Content", etc.)

#### 2. `content` (Place/Category)
**Definition**: Folders representing specific content themes, locations, or campaign/product names.
- **Examples**: "Zahuatil", "Hideouts & Lairs", "Gothic Cathedral", "Driftwood Frontier Town", "Ancora Bay"
- **Characteristics**: Thematic content containers, campaign settings, specific locations
- **Decision Rules**:
  - Contains multiple media types for same theme
  - Often child of creator folder
  - Names suggest places, campaigns, or content collections

#### 3. `media_bucket` (Media Type)
**Definition**: Folders organizing content by media format or delivery method.
- **Examples**: "VTT", "Print", "Maps", "Assets", "Tokens", "Music", "Illustrations"
- **Characteristics**: Technical organization, format-specific
- **Decision Rules**:
  - Name indicates media format or technical specification
  - Contains files of similar type/purpose
  - Often has uniform file extensions in descendants

#### 4. `descriptor` (Variant/Modifier)
**Definition**: Folders representing variations, conditions, or specific versions of content.
- **Examples**: "Clean", "Night", "Gridded", "Gridless", "Base", "Snow", "Ruins", "Animated Scenes"
- **Characteristics**: Modifies or qualifies parent content
- **Decision Rules**:
  - Name describes condition, time, style, or variation
  - Usually deeper in hierarchy (depth 3+)
  - Siblings often represent alternative versions

#### 5. `other`
**Definition**: Folders serving organizational or miscellaneous purposes.
- **Examples**: "Key & Design Notes", "Borough Guide", "Collaborator Content", "Bonus Content"
- **Characteristics**: Support materials, documentation, metadata
- **Decision Rules**:
  - Administrative or support function
  - Documentation, guides, or supplementary materials
  - Not primary content delivery

#### 6. `unknown`
**Definition**: Folders that cannot be confidently classified.
- **Use**: When folder purpose is ambiguous or insufficient context
- **Characteristics**: Unclear naming, insufficient hierarchical context

### Edge Cases and Ambiguities

1. **Creator vs Content Ambiguity**: Some folders like "Axziga's Lair - Collaboration with Tom Cartos" blend creator attribution with content naming.
   - **Solution**: If primary purpose is attribution (e.g., inside "Collaborator Content" folder), classify as `person_or_group`. If it's thematic content that happens to mention collaborator, classify as `content`.
   - **Key signal**: Sibling context - are siblings other creators or other content themes?

2. **Nested Creators**: Collaborators appear at various depths (e.g., "Borough Bound/Gruuk Jit'Jit/Collaborator Content/Cze and Peku - The Crystal Veil").
   - **Solution**: `person_or_group` applies to ANY creator attribution folder, regardless of depth. Use structural signals:
     - Parent context (is parent "Collaborator Content" or similar?)
     - Sibling uniformity (do siblings share similar attribution patterns?)
     - Content diversity (does folder contain varied media types?)

3. **Media vs Descriptor Overlap**: "VTT" could be media format or technical variant.
   - **Solution**: Prioritize `media_bucket` for technical formats, `descriptor` for aesthetic variants.
   - Use file extension uniformity and sibling context as disambiguating signals.

## 2. Training Data Strategy

### Sample Size Recommendations
- **Target**: 800-1,200 manually labeled samples (23-34% of 3,512 folders)
- **Minimum viable**: 600 samples (17% coverage)
- **Rationale**: Ensure 100+ examples per class while maintaining class balance

### Sampling Strategy

#### Depth Distribution
- **Depth 1**: 100% of samples (likely all creators, ~20-30 folders)
- **Depth 2**: 40% of samples (mix of creators and content, ~300-400 folders)
- **Depth 3**: 25% of samples (content and media buckets, ~400-500 folders)
- **Depth 4+**: 15% of samples (descriptors and variants, ~200-300 folders)

#### Diversity Requirements
1. **Creator Coverage**: Include samples from all major creators
2. **Hierarchical Diversity**: Ensure representation across folder tree structures
3. **Name Pattern Diversity**: Include various naming conventions and edge cases
4. **Sibling Context Diversity**: Sample folders with different sibling contexts

### Data Augmentation Strategies
1. **Synthetic Negative Mining**: Generate hard negatives by name similarity across classes
2. **Hierarchical Context Augmentation**: Use parent-child relationships for additional training signal
3. **Cross-Creator Pattern Mining**: Identify naming patterns that transcend creators

### Split Strategy
- **Train**: 70% (560-840 samples)
- **Validation**: 20% (160-240 samples)
- **Test**: 10% (80-120 samples)
- **Stratified by**: Label + Creator + Depth to ensure balanced representation

## 3. Feature Engineering Changes

### Enhanced Feature Extraction

#### Current Features to Retain
- **Hierarchical context**: grandparent/parent/self names (critical for classification)
- **Sibling context**: sibling names (helps distinguish organizational patterns)
- **Children context**: child names and counts (indicates folder purpose)
- **File extensions**: descendant extensions (strong signal for media_bucket)
- **Depth**: structural position (correlates with label types)

#### New Features to Add

1. **Structural Organization Signals** (replaces creator-specific features to avoid overfitting)
   ```
   content_diversity_score: float    # entropy of child folder types/names
   folder_to_file_ratio: float       # indicates organizational vs leaf container
   sibling_pattern_type: str         # detected sibling organization (creators/variants/media/mixed)
   parent_context_type: str          # detected parent organizational pattern
   has_attribution_parent: boolean   # parent is "Collaborator Content" or similar
   ```

2. **Media Type Signals**
   ```
   dominant_file_type: str           # most common descendant extension
   file_type_diversity: float        # entropy of file extensions
   has_uniform_media: boolean        # single media type vs mixed
   media_type_confidence: float      # strength of media type signal
   ```

3. **Naming Pattern Signals**
   ```
   name_pattern_type: str    # detected pattern (technical, aesthetic, location, etc.)
   has_collaboration_marker: boolean
   has_variant_keywords: boolean
   has_technical_keywords: boolean
   name_complexity: int      # word count, special chars
   ```

4. **Structural Signals**
   ```
   path_entropy: float       # diversity of ancestor names
   sibling_uniformity: float # how similar siblings are
   child_organization: str   # detected organization pattern
   is_leaf_container: boolean # has files but no folders
   ```

### Updated Feature Text Format
```
gp:{grandparent} | p:{parent} | self:{normalized_name} | depth:{depth} |
sibs:{sibling_summary} | sib_pattern:{sibling_pattern_type} |
children:{child_summary} | content_diversity:{diversity_score} | folder_file_ratio:{ratio} |
media:{dominant_type}({file_type_diversity}) | exts:{top_extensions} |
markers:collab={has_collab},variant={has_variant},tech={has_tech},attribution_ctx={has_attribution_parent} |
structure:path_entropy={path_entropy},sib_uniform={sibling_uniformity},leaf={is_leaf}
```

**Key changes from original proposal**:
- **REMOVED**: `creator:{creator_name}`, `creator_distance`, `is_known_creator` - avoids overfitting to specific creator names
- **ADDED**: `content_diversity`, `folder_file_ratio`, `sibling_pattern_type`, `has_attribution_parent` - generalizable structural patterns

## 4. Model & Training Approach

### Model Architecture Assessment

**SetFit Suitability**: ✅ **RECOMMENDED** - SetFit remains appropriate with modifications:
- Good for few-shot learning with limited labeled data
- Handles hierarchical text features well
- Triplet loss beneficial for distinguishing similar categories

### Recommended Approach: **Hierarchical SetFit**

#### Base Model Selection
- **Primary**: `sentence-transformers/all-MiniLM-L6-v2` (current choice - good balance)
- **Alternative**: `sentence-transformers/all-mpnet-base-v2` (if more capacity needed)
- **Domain-specific**: Consider fine-tuning on RPG/gaming text corpus first

#### Training Strategy Enhancements

1. **Hierarchical Loss Function**
   ```python
   # Combine triplet loss with hierarchical constraints
   class HierarchicalTripletLoss:
       def __init__(self, hierarchy_weight=0.3):
           self.triplet_loss = BatchHardSoftMarginTripletLoss()
           self.hierarchy_weight = hierarchy_weight

       def forward(self, embeddings, labels, metadata):
           triplet_loss = self.triplet_loss(embeddings, labels)
           hierarchy_loss = self.compute_hierarchy_loss(embeddings, metadata)
           return triplet_loss + self.hierarchy_weight * hierarchy_loss
   ```

2. **Class-Aware Sampling**
   - Ensure balanced representation in each batch
   - Oversample rare but important patterns (collaborations, edge cases)
   - Hard negative mining between confusable classes

3. **Multi-Task Learning** (Optional Enhancement)
   - Primary task: 6-way classification
   - Auxiliary task: Creator identification
   - Auxiliary task: Depth prediction
   - Shared encoder with task-specific heads

### Hyperparameter Recommendations
```python
TRAINING_CONFIG = {
    "batch_size": 32,  # Maintain for triplet loss
    "num_epochs": 8,   # Increase for complex task
    "learning_rate": 1e-5,  # Lower for stability
    "samples_per_label": 4,  # Increase for better triplet mining
    "hard_negative_factor": 3,  # More aggressive negative mining
    "hierarchy_weight": 0.2,  # If using hierarchical loss
}
```

## 5. Implementation Plan

### Phase 1: Core Infrastructure Updates (Priority 1)

1. **Update Label Definitions**
   ```python
   # training_utils.py
   VALID_LABELS = {
       "person_or_group",  # creator
       "content",          # place/category
       "media_bucket",     # media type
       "descriptor",       # variant/modifier
       "other",            # organizational
       "unknown"           # ambiguous
   }
   ```

2. **Enhance Feature Extraction** (`feature_extraction.py`)
   - Add creator detection logic
   - Add media type analysis
   - Add naming pattern detection
   - Update feature text format

3. **Update Sample Generation** (`training_utils.py`)
   - Modify sampling strategy for new depth distribution
   - Add creator-aware stratification
   - Update CSV schema for new features

### Phase 2: Training Pipeline Updates (Priority 2)

4. **Update Classification Pipeline** (`leaf_classifier.py`)
   - Update labels and hard negative configurations
   - Implement hierarchical loss if needed
   - Adjust hyperparameters

5. **Enhance Training Utilities**
   - Add hierarchical validation metrics
   - Implement creator-aware cross-validation
   - Add confusion matrix analysis tools

### Phase 3: Advanced Features (Priority 3)

6. **Add Hierarchical Constraints**
   - Parent-child label consistency checks
   - Path-based validation rules
   - Creator inheritance logic

7. **Implement Active Learning**
   - Uncertainty-based sample selection
   - Disagreement-based annotation prioritization
   - Iterative model improvement

### Database Schema Changes

**No schema changes required** - existing tables support new approach:
- `TrainingSample.label` field accommodates new labels
- Feature text stored in `text` field
- Additional metadata can use JSON fields

### Code Reuse Assessment

**High Reuse** (80%+):
- Database models and infrastructure
- Basic sampling and CSV generation
- SetFit training pipeline structure

**Moderate Changes** (modify existing):
- Feature extraction logic
- Label definitions and validation
- Training configuration

**New Implementation** (20%):
- Hierarchical loss functions (optional)
- Creator detection logic
- Advanced validation metrics

## 6. Validation & Iteration Strategy

### Classification Quality Validation

#### Core Metrics
1. **Per-Class Performance**
   - Precision, Recall, F1 per label
   - Class-specific confusion matrices
   - Support (sample count) per class

2. **Hierarchical Consistency**
   - Parent-child label compatibility rates
   - Creator inheritance accuracy
   - Path logic violation detection

3. **Error Analysis Metrics**
   - Most confused label pairs
   - Error patterns by creator/depth
   - Misclassification severity scoring

#### Validation Framework
```python
class HierarchicalValidator:
    def validate_labels(self, predictions, metadata):
        return {
            "accuracy": self.compute_accuracy(predictions),
            "per_class_f1": self.compute_per_class_f1(predictions),
            "hierarchy_violations": self.check_hierarchy_rules(predictions, metadata),
            "creator_consistency": self.check_creator_inheritance(predictions, metadata),
            "depth_appropriateness": self.check_depth_labels(predictions, metadata)
        }
```

### Active Learning Strategy

#### Uncertainty Sampling
1. **Prediction Confidence**: Target low-confidence predictions
2. **Class Boundary**: Sample near decision boundaries
3. **Hierarchical Disagreement**: Prioritize parent-child inconsistencies

#### Iterative Improvement Process
1. **Phase 1**: Label initial 600-800 samples
2. **Phase 2**: Train initial model, identify problematic patterns
3. **Phase 3**: Target additional 200-400 samples based on model weaknesses
4. **Phase 4**: Retrain and validate on holdout test set

#### Query Strategies
```python
def select_annotation_candidates(model, unlabeled_samples):
    candidates = []

    # Uncertainty sampling
    uncertainties = model.predict_proba(unlabeled_samples)
    uncertainty_scores = 1 - np.max(uncertainties, axis=1)

    # Diversity sampling
    embeddings = model.encode(unlabeled_samples)
    diversity_scores = compute_diversity_scores(embeddings, labeled_embeddings)

    # Hierarchical inconsistency
    hierarchy_scores = compute_hierarchy_inconsistency(unlabeled_samples)

    # Combine scores
    combined_scores = (0.4 * uncertainty_scores +
                      0.3 * diversity_scores +
                      0.3 * hierarchy_scores)

    return select_top_k(unlabeled_samples, combined_scores, k=50)
```

### Success Criteria

#### Minimum Viable Performance
- **Overall Accuracy**: >80%
- **Per-Class F1**: >70% for all classes
- **Hierarchy Violations**: <5%

#### Target Performance
- **Overall Accuracy**: >90%
- **Per-Class F1**: >85% for all classes
- **Hierarchy Violations**: <2%
- **Creator Consistency**: >95%

## Implementation Timeline

### Week 1-2: Infrastructure Setup
- Update label definitions and validation
- Implement enhanced feature extraction
- Generate new training samples CSV

### Week 3-4: Initial Training
- Manual labeling of core sample set (600-800 samples)
- Train initial model
- Validate and analyze performance

### Week 5-6: Refinement
- Implement active learning improvements
- Address identified weaknesses
- Retrain with expanded dataset

### Week 7-8: Production Integration
- Deploy classification pipeline
- Monitor performance on new data
- Iterate based on production feedback

## Risk Mitigation

### Technical Risks
- **Label Ambiguity**: Mitigate with clear decision rules and inter-annotator agreement testing
- **Class Imbalance**: Address through stratified sampling and class-aware training
- **Hierarchical Inconsistency**: Implement validation rules and hierarchical loss functions

### Data Quality Risks
- **Annotation Inconsistency**: Use multiple annotators and consensus labeling for difficult cases
- **Sampling Bias**: Ensure diverse representation across creators and folder structures
- **Edge Case Coverage**: Actively seek and include unusual naming patterns

This proposal provides a comprehensive roadmap for transitioning from the current author-focused classification system to a semantic folder-purpose classification system optimized for the Patreon RPG asset domain.