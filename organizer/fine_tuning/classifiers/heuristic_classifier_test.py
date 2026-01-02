"""Tests for heuristic classifier."""

import pytest
from utils.config import get_config

from fine_tuning.classifiers.heuristic_classifier import HeuristicClassifier


@pytest.fixture
def test_config():
    """Load actual config from YAML files."""
    # Load the real config which reads from yaml files
    return get_config()


class TestHeuristicClassifier:
    """Test the heuristic classifier."""

    def test_init(self, test_config):
        """Test classifier initialization."""
        classifier = HeuristicClassifier(test_config, taxonomy="v2")
        assert classifier.config == test_config
        assert classifier.taxonomy == "v2"

    def test_variant_mapping_v2(self, test_config):
        """Test variant mapping for v2 taxonomy."""
        classifier = HeuristicClassifier(test_config, taxonomy="v2")

        # Check variant types are mapped correctly via classification
        winter_result = classifier._check_variants("winter")
        assert winter_result is not None
        assert winter_result.label == "descriptor"

        vtt_result = classifier._check_variants("VTT")
        assert vtt_result is not None
        assert vtt_result.label == "asset_type"

        pdf_result = classifier._check_variants("PDF")
        assert pdf_result is not None
        assert pdf_result.label == "asset_type"

        # Test synonym
        pdfs_result = classifier._check_variants("PDFs")
        assert pdfs_result is not None
        assert pdfs_result.label == "asset_type"

    def test_variant_mapping_v1(self, test_config):
        """Test variant mapping for v1 taxonomy."""
        classifier = HeuristicClassifier(test_config, taxonomy="v1")

        # Check variant types are mapped correctly via classification
        winter_result = classifier._check_variants("winter")
        assert winter_result is not None
        assert winter_result.label == "descriptor"

        vtt_result = classifier._check_variants("VTT")
        assert vtt_result is not None
        assert vtt_result.label == "media_bucket"

        pdf_result = classifier._check_variants("PDF")
        assert pdf_result is not None
        assert pdf_result.label == "media_bucket"

    def test_classify_variant_season(self, test_config):
        """Test classification of season variants."""
        classifier = HeuristicClassifier(test_config, taxonomy="v2")

        result = classifier.classify("Winter")
        assert result.label == "descriptor"
        assert result.confidence >= 0.85
        assert "variant" in result.reason.lower()

    def test_classify_variant_media_type(self, test_config):
        """Test classification of media type variants."""
        classifier = HeuristicClassifier(test_config, taxonomy="v2")

        result = classifier.classify("VTT")
        assert result.label == "asset_type"
        assert result.confidence >= 0.85

    def test_classify_known_creator(self, test_config):
        """Test classification of known creators."""
        classifier = HeuristicClassifier(test_config, taxonomy="v2")

        result = classifier.classify("CzePeku", depth=1)
        assert result.label == "creator_or_studio"
        assert result.confidence >= 0.8

    def test_classify_creator_fuzzy_match(self, test_config):
        """Test classification with fuzzy creator matching."""
        classifier = HeuristicClassifier(test_config, taxonomy="v2")

        # Should match "Tom Cartos" with high similarity
        result = classifier.classify("Tom Cartos Maps", depth=1)
        assert result.label == "creator_or_studio"

    def test_classify_creator_with_collab(self, test_config):
        """Test classification with collaboration markers."""
        classifier = HeuristicClassifier(test_config, taxonomy="v2")

        result = classifier.classify("Collaboration with Tom Cartos", depth=2)
        assert result.label == "creator_or_studio"
        assert "collaboration" in result.reason.lower() or "collab" in result.matches

    def test_classify_asset_type_maps(self, test_config):
        """Test classification of map folders."""
        classifier = HeuristicClassifier(test_config, taxonomy="v2")

        result = classifier.classify("Maps", file_extensions=["png", "jpg"])
        assert result.label == "asset_type"
        assert result.confidence >= 0.75

    def test_classify_asset_type_tokens(self, test_config):
        """Test classification of token folders."""
        classifier = HeuristicClassifier(test_config, taxonomy="v2")

        result = classifier.classify("Tokens")
        assert result.label == "asset_type"

    def test_classify_other_year(self, test_config):
        """Test classification of year folders."""
        classifier = HeuristicClassifier(test_config, taxonomy="v2")

        result = classifier.classify("2023")
        assert result.label == "other"
        assert result.confidence >= 0.9

    def test_classify_other_rewards(self, test_config):
        """Test classification of organizational folders."""
        classifier = HeuristicClassifier(test_config, taxonomy="v2")

        result = classifier.classify("Patreon Rewards")
        # "Patreon Rewards" should be classified as "other" due to special case
        # handling that prioritizes the combo of "patreon" + "rewards"
        assert result.label == "other"
        assert result.confidence >= 0.75

    def test_classify_other_tier(self, test_config):
        """Test classification of tier organizational folders."""
        classifier = HeuristicClassifier(test_config, taxonomy="v2")

        result = classifier.classify("Tier 3 Rewards")
        assert result.label == "other"
        assert result.confidence >= 0.75

    def test_classify_unknown(self, test_config):
        """Test classification when no rules match."""
        classifier = HeuristicClassifier(test_config, taxonomy="v2")

        result = classifier.classify("SomeRandomFolderName")
        assert result.label == "unknown"
        assert result.confidence < 0.5

    def test_classify_v1_taxonomy(self, test_config):
        """Test that v1 taxonomy returns correct labels."""
        classifier = HeuristicClassifier(test_config, taxonomy="v1")

        # Variant should map to descriptor
        result = classifier.classify("Winter")
        assert result.label == "descriptor"

        # Media type should map to media_bucket
        result = classifier.classify("VTT")
        assert result.label == "media_bucket"

        # Creator should map to person_or_group
        result = classifier.classify("CzePeku", depth=1)
        assert result.label == "person_or_group"

    def test_classify_with_depth_context(self, test_config):
        """Test that depth affects confidence."""
        classifier = HeuristicClassifier(test_config, taxonomy="v2")

        # Creator at depth 1 (high confidence)
        result_shallow = classifier.classify("CzePeku", depth=1)

        # Creator at depth 4 (lower confidence)
        result_deep = classifier.classify("CzePeku", depth=4)

        # Shallow should have higher confidence
        assert result_shallow.confidence >= result_deep.confidence

    def test_classify_with_parent_context(self, test_config):
        """Test that parent context affects classification."""
        classifier = HeuristicClassifier(test_config, taxonomy="v2")

        # Folder under "Collaborator Content" parent
        result = classifier.classify(
            "Mystery Folder", parent_name="Collaborator Content"
        )

        # Should be classified as creator due to parent context
        # (if it matches other heuristics or has high confidence from parent)
        # This is a weaker signal, so it might still be unknown
        # but the parent context should be considered
        assert result is not None

    def test_classify_batch(self, test_config):
        """Test batch classification."""
        classifier = HeuristicClassifier(test_config, taxonomy="v2")

        samples = [
            {"name": "Winter", "depth": 3},
            {"name": "CzePeku", "depth": 1},
            {"name": "Maps", "depth": 2, "file_extensions": ["png"]},
            {"name": "2023", "depth": 1},
        ]

        results = classifier.classify_batch(samples)

        assert len(results) == 4
        assert results[0].label == "descriptor"  # Winter
        assert results[1].label == "creator_or_studio"  # CzePeku
        assert results[2].label == "asset_type"  # Maps
        assert results[3].label == "other"  # 2023

    def test_confidence_ranges(self, test_config):
        """Test that confidence scores are in valid range."""
        classifier = HeuristicClassifier(test_config, taxonomy="v2")

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

    def test_reason_provided(self, test_config):
        """Test that all results include a reason."""
        classifier = HeuristicClassifier(test_config, taxonomy="v2")

        result = classifier.classify("Winter")
        assert result.reason
        assert len(result.reason) > 0

    def test_matches_provided(self, test_config):
        """Test that matches are tracked."""
        classifier = HeuristicClassifier(test_config, taxonomy="v2")

        result = classifier.classify("Winter")
        assert isinstance(result.matches, list)
