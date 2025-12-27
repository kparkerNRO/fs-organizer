"""Unified taxonomy definitions and conversions for folder classification.

This module defines all label taxonomies and provides utilities for converting
between them. All taxonomy-related code should import from this module to ensure
consistency across the codebase.
"""

from __future__ import annotations

from typing import Dict, Set

# =============================================================================
# Taxonomy Definitions
# =============================================================================

# V1 Taxonomy (Original proposal)
LABELS_V1 = {
    "person_or_group",  # Creator, publisher, or studio
    "content",  # Specific subject matter or location
    "media_bucket",  # Type of media/asset
    "descriptor",  # Theme, variant, or modifier
    "other",  # Organizational/administrative folders
    "unknown",  # Cannot be confidently classified
}

# V2 Taxonomy (Refined proposal with clearer names)
LABELS_V2 = {
    "creator_or_studio",  # Creator, publisher, or studio (formerly person_or_group)
    "content_subject",  # Specific subject matter or location (formerly content)
    "asset_type",  # Type of media/asset (formerly media_bucket)
    "theme_or_genre",  # Theme, setting, genre, or variant (formerly descriptor)
    "other",  # Organizational/administrative folders
    "unknown",  # Cannot be confidently classified
}

# Legacy labels from existing training_utils code
# These are kept for backward compatibility with existing labeled data
LABELS_LEGACY = {
    "primary_author",  # Main creator
    "secondary_author",  # Collaborator or featured creator
    "collection",  # Named collection or series
    "subject",  # Subject matter or content type
    "media_format",  # File format grouping
    "media_type",  # Media type grouping
    "variant",  # Style, season, or other variant
    "other",  # Organizational folders
}

# =============================================================================
# Label Mapping / Conversion
# =============================================================================

# Map V1 labels to V2 labels (for upgrading)
V1_TO_V2: Dict[str, str] = {
    "person_or_group": "creator_or_studio",
    "content": "content_subject",
    "media_bucket": "asset_type",
    "descriptor": "theme_or_genre",
    "other": "other",
    "unknown": "unknown",
}

# Map V2 labels back to V1 (for downgrading)
V2_TO_V1: Dict[str, str] = {
    "creator_or_studio": "person_or_group",
    "content_subject": "content",
    "asset_type": "media_bucket",
    "theme_or_genre": "descriptor",
    "other": "other",
    "unknown": "unknown",
}

# Map legacy labels to V2 (best-effort mapping)
LEGACY_TO_V2: Dict[str, str] = {
    "primary_author": "creator_or_studio",
    "secondary_author": "creator_or_studio",
    "collection": "content_subject",
    "subject": "content_subject",
    "media_format": "asset_type",
    "media_type": "asset_type",
    "variant": "theme_or_genre",
    "other": "other",
}

# Map legacy labels to V1
LEGACY_TO_V1: Dict[str, str] = {
    "primary_author": "person_or_group",
    "secondary_author": "person_or_group",
    "collection": "content",
    "subject": "content",
    "media_format": "media_bucket",
    "media_type": "media_bucket",
    "variant": "descriptor",
    "other": "other",
}

# =============================================================================
# Variant Type to Taxonomy Mapping
# =============================================================================

# Mapping from variant types (in variants.yaml) to taxonomy labels
# Format: {variant_type: (v1_label, v2_label)}
VARIANT_TYPE_TO_TAXONOMY: Dict[str, tuple[str, str]] = {
    "variant": ("descriptor", "theme_or_genre"),
    "media_type": ("media_bucket", "asset_type"),
    "media_format": ("media_bucket", "asset_type"),
}


def build_variant_mappings(variants: Dict[str, Dict[str, any]]) -> tuple[Dict[str, str], Dict[str, str]]:
    """Build mappings from config variants to taxonomy labels.

    Args:
        variants: Variant definitions from config (e.g., config.variants)

    Returns:
        Tuple of (variant_to_label_v1, variant_to_label_v2) mappings
    """
    variant_to_label_v1: Dict[str, str] = {}
    variant_to_label_v2: Dict[str, str] = {}

    # Map each variant name to its label based on type
    for variant_name, variant_info in variants.items():
        variant_type = variant_info.get("type", "variant")

        if variant_type in VARIANT_TYPE_TO_TAXONOMY:
            v1_label, v2_label = VARIANT_TYPE_TO_TAXONOMY[variant_type]
            variant_to_label_v1[variant_name] = v1_label
            variant_to_label_v2[variant_name] = v2_label

            # Also map synonyms
            for synonym in variant_info.get("synonyms", []):
                variant_to_label_v1[synonym] = v1_label
                variant_to_label_v2[synonym] = v2_label

    return variant_to_label_v1, variant_to_label_v2


# =============================================================================
# Utility Functions
# =============================================================================

def get_labels(taxonomy: str) -> Set[str]:
    """Get label set for a specific taxonomy.

    Args:
        taxonomy: Taxonomy name ('v1', 'v2', or 'legacy')

    Returns:
        Set of valid labels for that taxonomy

    Raises:
        ValueError: If taxonomy is not recognized
    """
    if taxonomy == "v1":
        return LABELS_V1
    elif taxonomy == "v2":
        return LABELS_V2
    elif taxonomy == "legacy":
        return LABELS_LEGACY
    else:
        raise ValueError(f"Unknown taxonomy: {taxonomy}. Must be 'v1', 'v2', or 'legacy'")


def convert_label(label: str, from_taxonomy: str, to_taxonomy: str) -> str:
    """Convert a label from one taxonomy to another.

    Args:
        label: Label to convert
        from_taxonomy: Source taxonomy ('v1', 'v2', or 'legacy')
        to_taxonomy: Target taxonomy ('v1', 'v2', or 'legacy')

    Returns:
        Converted label, or original label if no mapping exists

    Raises:
        ValueError: If taxonomy names are not recognized
    """
    # Validate taxonomies
    get_labels(from_taxonomy)  # Raises ValueError if invalid
    get_labels(to_taxonomy)

    # No conversion needed
    if from_taxonomy == to_taxonomy:
        return label

    # Select appropriate mapping
    if from_taxonomy == "v1" and to_taxonomy == "v2":
        return V1_TO_V2.get(label, label)
    elif from_taxonomy == "v2" and to_taxonomy == "v1":
        return V2_TO_V1.get(label, label)
    elif from_taxonomy == "legacy" and to_taxonomy == "v2":
        return LEGACY_TO_V2.get(label, label)
    elif from_taxonomy == "legacy" and to_taxonomy == "v1":
        return LEGACY_TO_V1.get(label, label)
    elif from_taxonomy == "v1" and to_taxonomy == "legacy":
        # Reverse lookup in LEGACY_TO_V1
        for legacy_label, v1_label in LEGACY_TO_V1.items():
            if v1_label == label:
                return legacy_label
        return label
    elif from_taxonomy == "v2" and to_taxonomy == "legacy":
        # Reverse lookup in LEGACY_TO_V2
        for legacy_label, v2_label in LEGACY_TO_V2.items():
            if v2_label == label:
                return legacy_label
        return label
    else:
        # Shouldn't reach here if validation works
        return label


def is_valid_label(label: str, taxonomy: str) -> bool:
    """Check if a label is valid for a given taxonomy.

    Args:
        label: Label to check
        taxonomy: Taxonomy name ('v1', 'v2', or 'legacy')

    Returns:
        True if label is valid for the taxonomy

    Raises:
        ValueError: If taxonomy is not recognized
    """
    labels = get_labels(taxonomy)
    return label in labels


def normalize_labels(labels: Set[str], target_taxonomy: str = "v2") -> Set[str]:
    """Normalize a set of labels to a target taxonomy.

    Attempts to detect the source taxonomy and convert all labels.

    Args:
        labels: Set of labels to normalize
        target_taxonomy: Target taxonomy ('v1', 'v2', or 'legacy')

    Returns:
        Set of normalized labels
    """
    # Try to detect source taxonomy
    source_taxonomy = None

    # Check if all labels are in V1
    if labels.issubset(LABELS_V1):
        source_taxonomy = "v1"
    # Check if all labels are in V2
    elif labels.issubset(LABELS_V2):
        source_taxonomy = "v2"
    # Check if all labels are in legacy
    elif labels.issubset(LABELS_LEGACY):
        source_taxonomy = "legacy"

    # If we detected a source taxonomy, convert
    if source_taxonomy and source_taxonomy != target_taxonomy:
        return {convert_label(label, source_taxonomy, target_taxonomy) for label in labels}

    # Otherwise, return as-is
    return labels
