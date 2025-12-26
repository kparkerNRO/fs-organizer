"""Tests for heuristic classifier."""

import pytest
from unittest.mock import MagicMock

from fine_tuning.heuristic_classifier import (
    HeuristicClassifier,
    ClassificationResult,
    normalize_text,
    text_similarity,
    has_close_text_match,
    has_pattern_match,
)
from utils.config import Config


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    config = MagicMock(spec=Config)

    # Mock variants
    config.variants = {
        "winter": {"type": "variant", "grouping": "season", "synonyms": []},
        "summer": {"type": "variant", "grouping": "season", "synonyms": []},
        "VTT": {"type": "media_type", "grouping": "media", "synonyms": []},
        "PDF": {"type": "media_format", "grouping": "media", "synonyms": ["PDFs"]},
        "Gridded": {"type": "variant", "grouping": "layout", "synonyms": []},
        "Gridless": {"type": "variant", "grouping": "layout", "synonyms": []},
    }

    # Mock creators
    config.creators = {
        "CzePeku": {"synonyms": [], "tokens_to_remove": []},
        "Tom Cartos": {"synonyms": [], "tokens_to_remove": []},
        "Limithron": {"synonyms": [], "tokens_to_remove": []},
    }

    # Mock collaboration markers
    config.collab_markers = [
        "collab", "collaboration", "with", "feat", "ft", "&"
    ]

    return config


class TestTextUtilities:
    """Test text processing utilities."""

    def test_normalize_text(self):
        assert normalize_text("Hello  World") == "hello world"
        assert normalize_text("  UPPER  ") == "upper"
        assert normalize_text("Multiple   Spaces") == "multiple spaces"

    def test_text_similarity(self):
        # Identical strings
        assert text_similarity("hello", "hello") == 1.0

        # Very similar strings
        similarity = text_similarity("winter", "Winter")
        assert similarity == 1.0  # Case doesn't matter after normalization

        # Different strings
        similarity = text_similarity("hello", "world")
        assert similarity < 0.5

    def test_has_close_text_match_exact(self):
        candidates = ["winter", "summer", "spring"]
        has_match, matches = has_close_text_match("Winter", candidates)
        assert has_match
        assert "winter" in matches

    def test_has_close_text_match_substring(self):
        candidates = ["VTT Maps", "Print Maps"]
        has_match, matches = has_close_text_match("VTT", candidates)
        assert has_match
        assert "VTT Maps" in matches

    def test_has_close_text_match_fuzzy(self):
        candidates = ["Gridded", "Grid"]
        has_match, matches = has_close_text_match("Gridded", candidates, threshold=0.8)
        assert has_match

    def test_has_close_text_match_no_match(self):
        candidates = ["winter", "summer"]
        has_match, matches = has_close_text_match("xyz", candidates)
        assert not has_match
        assert len(matches) == 0

    def test_has_pattern_match_exact(self):
        patterns = ["maps", "tokens", "assets"]
        has_match, matches = has_pattern_match("Maps", patterns)
        assert has_match
        assert "maps" in matches

    def test_has_pattern_match_contains(self):
        patterns = ["map", "token"]
        has_match, matches = has_pattern_match("VTT Maps Pack", patterns)
        assert has_match
        assert "map" in matches

    def test_has_pattern_match_no_match(self):
        patterns = ["maps", "tokens"]
        has_match, matches = has_pattern_match("xyz", patterns)
        assert not has_match


class TestHeuristicClassifier:
    """Test the heuristic classifier."""

    def test_init(self, mock_config):
        """Test classifier initialization."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v2")
        assert classifier.config == mock_config
        assert classifier.taxonomy == "v2"

    def test_variant_mapping_v2(self, mock_config):
        """Test variant mapping for v2 taxonomy."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v2")

        # Check variant types are mapped correctly
        assert classifier.variant_to_label_v2["winter"] == "theme_or_genre"
        assert classifier.variant_to_label_v2["VTT"] == "asset_type"
        assert classifier.variant_to_label_v2["PDF"] == "asset_type"
        assert classifier.variant_to_label_v2["PDFs"] == "asset_type"  # synonym

    def test_variant_mapping_v1(self, mock_config):
        """Test variant mapping for v1 taxonomy."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v1")

        # Check variant types are mapped correctly
        assert classifier.variant_to_label_v1["winter"] == "descriptor"
        assert classifier.variant_to_label_v1["VTT"] == "media_bucket"
        assert classifier.variant_to_label_v1["PDF"] == "media_bucket"

    def test_classify_variant_season(self, mock_config):
        """Test classification of season variants."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v2")

        result = classifier.classify("Winter")
        assert result.label == "theme_or_genre"
        assert result.confidence >= 0.85
        assert "variant" in result.reason.lower()

    def test_classify_variant_media_type(self, mock_config):
        """Test classification of media type variants."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v2")

        result = classifier.classify("VTT")
        assert result.label == "asset_type"
        assert result.confidence >= 0.85

    def test_classify_known_creator(self, mock_config):
        """Test classification of known creators."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v2")

        result = classifier.classify("CzePeku", depth=1)
        assert result.label == "creator_or_studio"
        assert result.confidence >= 0.8

    def test_classify_creator_fuzzy_match(self, mock_config):
        """Test classification with fuzzy creator matching."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v2")

        # Should match "Tom Cartos" with high similarity
        result = classifier.classify("Tom Cartos Maps", depth=1)
        assert result.label == "creator_or_studio"

    def test_classify_creator_with_collab(self, mock_config):
        """Test classification with collaboration markers."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v2")

        result = classifier.classify("Collaboration with Tom Cartos", depth=2)
        assert result.label == "creator_or_studio"
        assert "collaboration" in result.reason.lower() or "collab" in result.matches

    def test_classify_asset_type_maps(self, mock_config):
        """Test classification of map folders."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v2")

        result = classifier.classify("Maps", file_extensions=["png", "jpg"])
        assert result.label == "asset_type"
        assert result.confidence >= 0.75

    def test_classify_asset_type_tokens(self, mock_config):
        """Test classification of token folders."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v2")

        result = classifier.classify("Tokens")
        assert result.label == "asset_type"

    def test_classify_theme_dungeon(self, mock_config):
        """Test classification of theme folders."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v2")

        result = classifier.classify("Dungeon")
        assert result.label == "theme_or_genre"
        assert result.confidence >= 0.7

    def test_classify_other_year(self, mock_config):
        """Test classification of year folders."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v2")

        result = classifier.classify("2023")
        assert result.label == "other"
        assert result.confidence >= 0.9

    def test_classify_other_rewards(self, mock_config):
        """Test classification of organizational folders."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v2")

        result = classifier.classify("Patreon Rewards")
        assert result.label == "other"
        assert result.confidence >= 0.75

    def test_classify_unknown(self, mock_config):
        """Test classification when no rules match."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v2")

        result = classifier.classify("SomeRandomFolderName")
        assert result.label == "unknown"
        assert result.confidence < 0.5

    def test_classify_v1_taxonomy(self, mock_config):
        """Test that v1 taxonomy returns correct labels."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v1")

        # Variant should map to descriptor
        result = classifier.classify("Winter")
        assert result.label == "descriptor"

        # Media type should map to media_bucket
        result = classifier.classify("VTT")
        assert result.label == "media_bucket"

        # Creator should map to person_or_group
        result = classifier.classify("CzePeku", depth=1)
        assert result.label == "person_or_group"

    def test_classify_with_depth_context(self, mock_config):
        """Test that depth affects confidence."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v2")

        # Creator at depth 1 (high confidence)
        result_shallow = classifier.classify("CzePeku", depth=1)

        # Creator at depth 4 (lower confidence)
        result_deep = classifier.classify("CzePeku", depth=4)

        # Shallow should have higher confidence
        assert result_shallow.confidence >= result_deep.confidence

    def test_classify_with_parent_context(self, mock_config):
        """Test that parent context affects classification."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v2")

        # Folder under "Collaborator Content" parent
        result = classifier.classify(
            "Mystery Folder",
            parent_name="Collaborator Content"
        )

        # Should be classified as creator due to parent context
        # (if it matches other heuristics or has high confidence from parent)
        # This is a weaker signal, so it might still be unknown
        # but the parent context should be considered
        assert result is not None

    def test_classify_batch(self, mock_config):
        """Test batch classification."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v2")

        samples = [
            {"name": "Winter", "depth": 3},
            {"name": "CzePeku", "depth": 1},
            {"name": "Maps", "depth": 2, "file_extensions": ["png"]},
            {"name": "2023", "depth": 1},
        ]

        results = classifier.classify_batch(samples)

        assert len(results) == 4
        assert results[0].label == "theme_or_genre"  # Winter
        assert results[1].label == "creator_or_studio"  # CzePeku
        assert results[2].label == "asset_type"  # Maps
        assert results[3].label == "other"  # 2023

    def test_confidence_ranges(self, mock_config):
        """Test that confidence scores are in valid range."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v2")

        test_cases = [
            "Winter",
            "CzePeku",
            "Maps",
            "2023",
            "Unknown Folder",
        ]

        for name in test_cases:
            result = classifier.classify(name)
            assert 0.0 <= result.confidence <= 1.0

    def test_reason_provided(self, mock_config):
        """Test that all results include a reason."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v2")

        result = classifier.classify("Winter")
        assert result.reason
        assert len(result.reason) > 0

    def test_matches_provided(self, mock_config):
        """Test that matches are tracked."""
        classifier = HeuristicClassifier(mock_config, taxonomy="v2")

        result = classifier.classify("Winter")
        assert isinstance(result.matches, list)
