"""Heuristic classifier for folder classification.

This module implements a rule-based classifier using patterns from the config files
and proposals to assign initial labels with confidence scores.

The classifier supports both v1 and v2 taxonomies and provides confidence scores
to facilitate human review of the labeling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

from utils.config import Config


# Label taxonomies from the proposals
LABELS_V1 = {
    "person_or_group",  # creator
    "content",  # place/category
    "media_bucket",  # media type
    "descriptor",  # variant/modifier
    "other",  # organizational
    "unknown",  # ambiguous
}

LABELS_V2 = {
    "creator_or_studio",  # formerly person_or_group
    "content_subject",  # formerly content
    "theme_or_genre",  # formerly descriptor
    "asset_type",  # formerly media_bucket
    "other",  # organizational
    "unknown",  # ambiguous
}


@dataclass
class ClassificationResult:
    """Result of heuristic classification."""

    label: str
    confidence: float
    reason: str
    matches: List[str]  # What patterns/rules matched


# Mapping from variant types (in variants.yaml) to taxonomy labels
# Format: {variant_type: (v1_label, v2_label)}
VARIANT_TYPE_TO_TAXONOMY = {
    "variant": ("descriptor", "theme_or_genre"),
    "media_type": ("media_bucket", "asset_type"),
    "media_format": ("media_bucket", "asset_type"),
}


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove extra spaces)."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using SequenceMatcher.

    Returns a value between 0 and 1, where 1 is identical.
    """
    return SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio()


def has_close_text_match(target: str, candidates: List[str], threshold: float = 0.85) -> Tuple[bool, List[str]]:
    """Check if target has a close text match in candidates.

    Args:
        target: Text to match
        candidates: List of candidate strings
        threshold: Similarity threshold (0-1)

    Returns:
        Tuple of (has_match, list_of_matches)
    """
    matches = []
    normalized_target = normalize_text(target)

    for candidate in candidates:
        normalized_candidate = normalize_text(candidate)

        # Exact match
        if normalized_target == normalized_candidate:
            matches.append(candidate)
            continue

        # Substring match
        if normalized_target in normalized_candidate or normalized_candidate in normalized_target:
            matches.append(candidate)
            continue

        # Fuzzy match
        similarity = text_similarity(target, candidate)
        if similarity >= threshold:
            matches.append(candidate)

    return len(matches) > 0, matches


def has_pattern_match(text: str, patterns: List[str]) -> Tuple[bool, List[str]]:
    """Check if text matches any of the given patterns (case-insensitive).

    Args:
        text: Text to check
        patterns: List of patterns (can include wildcards)

    Returns:
        Tuple of (has_match, list_of_matching_patterns)
    """
    matches = []
    normalized_text = normalize_text(text)

    for pattern in patterns:
        normalized_pattern = normalize_text(pattern)

        # Exact match
        if normalized_text == normalized_pattern:
            matches.append(pattern)
            continue

        # Contains match (pattern in text)
        if normalized_pattern in normalized_text:
            matches.append(pattern)

    return len(matches) > 0, matches


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
        self._build_creator_patterns()
        self._build_asset_type_patterns()
        self._build_theme_patterns()

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

    def _build_creator_patterns(self):
        """Build patterns for creator detection."""
        # Known creators from config
        self.known_creators = set(self.config.creators.keys())

        # Patterns that suggest creator/studio
        self.creator_keywords = {
            "patreon", "studios", "studio", "co", "inc", "llc",
            "cartographer", "maps", "publishing", "games"
        }

        # Collaboration markers
        self.collab_markers = set(self.config.collab_markers)

    def _build_asset_type_patterns(self):
        """Build patterns for asset type detection."""
        # From proposal v2
        self.asset_type_keywords = {
            "maps", "tokens", "token", "pack", "assets", "asset",
            "battlemap", "battlemaps", "handouts", "handout",
            "tiles", "tile", "music", "illustrations", "illustration",
            "animated scenes", "key & design notes"
        }

    def _build_theme_patterns(self):
        """Build patterns for theme/genre detection."""
        # From proposal v2
        self.theme_keywords = {
            "dungeon", "dungeons", "forest", "sci-fi", "scifi",
            "cyberpunk", "horror", "desert", "urban", "fantasy",
            "medieval", "modern", "futuristic", "gothic", "steampunk"
        }

        # Year patterns for "other"
        self.year_pattern = re.compile(r'^(19|20)\d{2}$')

        # Other organizational keywords
        self.other_keywords = {
            "rewards", "reward", "tier", "bonus", "content",
            "instructions", "instruction", "guide", "notes", "note"
        }

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
        """Get the label for the current taxonomy."""
        # Map v1 labels to v2 if needed
        if self.taxonomy == "v2":
            label_map = {
                "person_or_group": "creator_or_studio",
                "content": "content_subject",
                "media_bucket": "asset_type",
                "descriptor": "theme_or_genre",
            }
            return label_map.get(base_label, base_label)
        return base_label

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
        has_match, creator_matches = has_close_text_match(name, list(self.known_creators), threshold=0.85)
        if has_match:
            matches.extend(creator_matches)

        # Check for creator keywords
        has_keyword, keyword_matches = has_pattern_match(name, list(self.creator_keywords))
        if has_keyword:
            matches.extend(keyword_matches)

        # Check for collaboration markers
        has_collab, collab_matches = has_pattern_match(name, list(self.collab_markers))
        if has_collab:
            matches.append("collaboration marker")

        # Check if parent suggests creator context
        if parent_name:
            parent_lower = normalize_text(parent_name)
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
        has_match, matches = has_pattern_match(name, list(self.asset_type_keywords))

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
        has_match, matches = has_pattern_match(name, list(self.theme_keywords))

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
        if self.year_pattern.match(name.strip()):
            return ClassificationResult(
                label=self._get_label("other"),
                confidence=0.95,
                reason="Year folder",
                matches=[name]
            )

        # Check for other organizational keywords
        has_match, matches = has_pattern_match(name, list(self.other_keywords))

        if has_match:
            # Special case: if contains both "patreon" and reward/tier keywords, higher confidence
            # This handles "Patreon Rewards", "Patreon Tier 1", etc.
            normalized_name = normalize_text(name)
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
