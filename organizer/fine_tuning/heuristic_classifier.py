"""Heuristic classifier for folder classification.

This module implements a rule-based classifier using patterns from the config files
and proposals to assign initial labels with confidence scores.

The classifier supports both v1 and v2 taxonomies and provides confidence scores
to facilitate human review of the labeling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from fine_tuning.taxonomy import get_labels, convert_label
from fine_tuning.text_processing import (
    has_close_text_match,
    has_pattern_match,
    normalize_string,
)
from utils.config import Config


# Mapping from variant types (in variants.yaml) to taxonomy labels
# Format: {variant_type: (v1_label, v2_label)}
VARIANT_TYPE_TO_TAXONOMY = {
    "variant": ("descriptor", "theme_or_genre"),
    "media_type": ("media_bucket", "asset_type"),
    "media_format": ("media_bucket", "asset_type"),
}

# Creator detection keywords (from proposal v2)
CREATOR_KEYWORDS = {
    "patreon", "studios", "studio", "co", "inc", "llc",
    "cartographer", "maps", "publishing", "games"
}

# Asset type keywords (from proposal v2)
ASSET_TYPE_KEYWORDS = {
    "maps", "tokens", "token", "pack", "assets", "asset",
    "battlemap", "battlemaps", "handouts", "handout",
    "tiles", "tile", "music", "illustrations", "illustration",
    "animated scenes", "key & design notes"
}

# Theme/genre keywords (from proposal v2)
THEME_KEYWORDS = {
    "dungeon", "dungeons", "forest", "sci-fi", "scifi",
    "cyberpunk", "horror", "desert", "urban", "fantasy",
    "medieval", "modern", "futuristic", "gothic", "steampunk"
}

# Organizational folder keywords
OTHER_KEYWORDS = {
    "rewards", "reward", "tier", "bonus", "content",
    "instructions", "instruction", "guide", "notes", "note"
}

# Year pattern for organizational folders
YEAR_PATTERN = re.compile(r'^(19|20)\d{2}$')


@dataclass
class ClassificationResult:
    """Result of heuristic classification."""

    label: str
    confidence: float
    reason: str
    matches: List[str]  # What patterns/rules matched


class HeuristicClassifier:
    """Rule-based classifier using config and proposal heuristics."""

    def __init__(self, config: Config, taxonomy: str = "v2"):
        """Initialize the heuristic classifier.

        Args:
            config: Configuration with variants and creators
            taxonomy: Which taxonomy to use ('v1' or 'v2')
        """
        self.config = config
        self.taxonomy = taxonomy

        # Build lookup tables from config
        self._build_variant_mappings()

    def _build_variant_mappings(self):
        """Build mappings from config variants to taxonomy labels."""
        self.variant_to_label_v1: Dict[str, str] = {}
        self.variant_to_label_v2: Dict[str, str] = {}

        # Map each variant name to its label based on type
        for variant_name, variant_info in self.config.variants.items():
            variant_type = variant_info.get("type", "variant")

            if variant_type in VARIANT_TYPE_TO_TAXONOMY:
                v1_label, v2_label = VARIANT_TYPE_TO_TAXONOMY[variant_type]
                self.variant_to_label_v1[variant_name] = v1_label
                self.variant_to_label_v2[variant_name] = v2_label

                # Also map synonyms
                for synonym in variant_info.get("synonyms", []):
                    self.variant_to_label_v1[synonym] = v1_label
                    self.variant_to_label_v2[synonym] = v2_label

    def classify(self, name: str, depth: int = 0, parent_name: Optional[str] = None,
                 children_names: Optional[List[str]] = None,
                 sibling_names: Optional[List[str]] = None,
                 file_extensions: Optional[List[str]] = None) -> ClassificationResult:
        """Classify a folder using heuristic rules.

        Args:
            name: Folder name
            depth: Depth in folder hierarchy
            parent_name: Parent folder name
            children_names: Names of child folders
            sibling_names: Names of sibling folders
            file_extensions: File extensions in folder

        Returns:
            ClassificationResult with label, confidence, and reasoning
        """
        children_names = children_names or []
        sibling_names = sibling_names or []
        file_extensions = file_extensions or []

        # Try classification rules in priority order
        results = []

        # 1. Check for known variants (high confidence)
        result = self._check_variants(name)
        if result:
            results.append(result)

        # 2. Check for "other" category first (some have very high confidence like years)
        # This needs to be before creator check to avoid "Patreon Rewards" being classified as creator
        result = self._check_other(name)
        if result:
            results.append(result)

        # 3. Check for creators (high confidence at low depth)
        result = self._check_creators(name, depth, parent_name, children_names, sibling_names)
        if result:
            results.append(result)

        # 4. Check for asset types (medium confidence)
        result = self._check_asset_types(name, file_extensions)
        if result:
            results.append(result)

        # 5. Check for themes (medium confidence)
        result = self._check_themes(name)
        if result:
            results.append(result)

        # Return highest confidence result, or unknown
        if results:
            # Sort by confidence (descending)
            results.sort(key=lambda r: r.confidence, reverse=True)
            return results[0]

        # Default to unknown with low confidence
        return ClassificationResult(
            label=self._get_label("unknown"),
            confidence=0.3,
            reason="No matching heuristic rules",
            matches=[]
        )

    def _get_label(self, base_label: str) -> str:
        """Get the label for the current taxonomy.

        Args:
            base_label: Base label in V1 taxonomy

        Returns:
            Label converted to current taxonomy
        """
        if self.taxonomy == "v1":
            return base_label
        else:
            # Convert from v1 to target taxonomy
            return convert_label(base_label, "v1", self.taxonomy)

    def _check_variants(self, name: str) -> Optional[ClassificationResult]:
        """Check if name matches known variants from config."""
        variant_map = self.variant_to_label_v2 if self.taxonomy == "v2" else self.variant_to_label_v1

        # Get all variant names
        variant_names = list(variant_map.keys())

        # Check for close text match
        has_match, matches = has_close_text_match(name, variant_names, threshold=0.85)

        if has_match and matches:
            # Use the first match to get the label
            label = variant_map[matches[0]]
            return ClassificationResult(
                label=label,
                confidence=0.9,  # High confidence for config-based matches
                reason=f"Matched variant '{matches[0]}' from config",
                matches=matches
            )

        return None

    def _check_creators(self, name: str, depth: int, parent_name: Optional[str],
                       children_names: List[str], sibling_names: List[str]) -> Optional[ClassificationResult]:
        """Check if folder represents a creator/studio."""
        matches = []

        # Check against known creators
        known_creators = list(self.config.creators.keys())
        has_match, creator_matches = has_close_text_match(name, known_creators, threshold=0.85)
        if has_match:
            matches.extend(creator_matches)

        # Check for creator keywords
        has_keyword, keyword_matches = has_pattern_match(name, list(CREATOR_KEYWORDS))
        if has_keyword:
            matches.extend(keyword_matches)

        # Check for collaboration markers
        collab_markers = list(self.config.collab_markers)
        has_collab, collab_matches = has_pattern_match(name, collab_markers)
        if has_collab:
            matches.append("collaboration marker")

        # Check if parent suggests creator context
        if parent_name:
            parent_lower = normalize_string(parent_name)
            if "collaborat" in parent_lower or "creator" in parent_lower or "author" in parent_lower:
                matches.append("creator context from parent")

        if matches:
            # Higher confidence at shallower depths
            base_confidence = 0.85 if depth <= 2 else 0.7

            # Boost confidence if multiple signals
            if len(matches) >= 2:
                base_confidence = min(0.95, base_confidence + 0.1)

            return ClassificationResult(
                label=self._get_label("person_or_group"),
                confidence=base_confidence,
                reason=f"Creator indicators: {', '.join(matches[:3])}",
                matches=matches
            )

        return None

    def _check_asset_types(self, name: str, file_extensions: List[str]) -> Optional[ClassificationResult]:
        """Check if folder represents an asset type."""
        has_match, matches = has_pattern_match(name, list(ASSET_TYPE_KEYWORDS))

        if has_match:
            # Check file extensions for additional confidence
            confidence = 0.8
            if file_extensions:
                # Boost confidence if files are present
                confidence = min(0.9, confidence + 0.1)

            return ClassificationResult(
                label=self._get_label("media_bucket"),
                confidence=confidence,
                reason=f"Matched asset type keywords: {', '.join(matches[:3])}",
                matches=matches
            )

        return None

    def _check_themes(self, name: str) -> Optional[ClassificationResult]:
        """Check if folder represents a theme/genre."""
        has_match, matches = has_pattern_match(name, list(THEME_KEYWORDS))

        if has_match:
            return ClassificationResult(
                label=self._get_label("descriptor"),
                confidence=0.75,
                reason=f"Matched theme keywords: {', '.join(matches[:3])}",
                matches=matches
            )

        return None

    def _check_other(self, name: str) -> Optional[ClassificationResult]:
        """Check if folder is in 'other' category."""
        # Check for year
        if YEAR_PATTERN.match(name.strip()):
            return ClassificationResult(
                label=self._get_label("other"),
                confidence=0.95,
                reason="Year folder",
                matches=[name]
            )

        # Check for other organizational keywords
        has_match, matches = has_pattern_match(name, list(OTHER_KEYWORDS))

        if has_match:
            # Special case: if contains both "patreon" and reward/tier keywords, higher confidence
            # This handles "Patreon Rewards", "Patreon Tier 1", etc.
            normalized_name = normalize_string(name)
            if "patreon" in normalized_name and any(kw in normalized_name for kw in ["reward", "tier", "bonus"]):
                return ClassificationResult(
                    label=self._get_label("other"),
                    confidence=0.90,  # Higher confidence to beat creator check
                    reason=f"Patreon organizational folder: {', '.join(matches[:3])}",
                    matches=matches
                )

            return ClassificationResult(
                label=self._get_label("other"),
                confidence=0.8,
                reason=f"Organizational keywords: {', '.join(matches[:3])}",
                matches=matches
            )

        return None

    def classify_batch(self, samples: List[Dict]) -> List[ClassificationResult]:
        """Classify multiple samples.

        Args:
            samples: List of dicts with keys: name, depth, parent_name, etc.

        Returns:
            List of ClassificationResult objects
        """
        results = []

        for sample in samples:
            result = self.classify(
                name=sample.get("name", ""),
                depth=sample.get("depth", 0),
                parent_name=sample.get("parent_name"),
                children_names=sample.get("children_names"),
                sibling_names=sample.get("sibling_names"),
                file_extensions=sample.get("file_extensions")
            )
            results.append(result)

        return results
