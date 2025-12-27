"""Tests for taxonomy module."""

import pytest

from fine_tuning.taxonomy import (
    LABELS_V1,
    LABELS_V2,
    LABELS_LEGACY,
    V1_TO_V2,
    V2_TO_V1,
    LEGACY_TO_V2,
    LEGACY_TO_V1,
    VARIANT_TYPE_TO_TAXONOMY,
    get_labels,
    convert_label,
    is_valid_label,
    normalize_labels,
    build_variant_mappings,
)


class TestLabelSets:
    """Test that label sets are properly defined."""

    def test_v1_labels_exist(self):
        assert len(LABELS_V1) == 6
        assert "person_or_group" in LABELS_V1
        assert "content" in LABELS_V1
        assert "media_bucket" in LABELS_V1
        assert "descriptor" in LABELS_V1
        assert "other" in LABELS_V1
        assert "unknown" in LABELS_V1

    def test_v2_labels_exist(self):
        assert len(LABELS_V2) == 6
        assert "creator_or_studio" in LABELS_V2
        assert "content_subject" in LABELS_V2
        assert "asset_type" in LABELS_V2
        assert "theme_or_genre" in LABELS_V2
        assert "other" in LABELS_V2
        assert "unknown" in LABELS_V2

    def test_legacy_labels_exist(self):
        assert len(LABELS_LEGACY) == 8
        assert "primary_author" in LABELS_LEGACY
        assert "secondary_author" in LABELS_LEGACY
        assert "collection" in LABELS_LEGACY
        assert "subject" in LABELS_LEGACY
        assert "media_format" in LABELS_LEGACY
        assert "media_type" in LABELS_LEGACY
        assert "variant" in LABELS_LEGACY
        assert "other" in LABELS_LEGACY


class TestMappings:
    """Test that conversion mappings are complete."""

    def test_v1_to_v2_complete(self):
        """All V1 labels should have V2 mappings."""
        for label in LABELS_V1:
            assert label in V1_TO_V2

    def test_v2_to_v1_complete(self):
        """All V2 labels should have V1 mappings."""
        for label in LABELS_V2:
            assert label in V2_TO_V1

    def test_v1_v2_roundtrip(self):
        """V1 -> V2 -> V1 should return original label."""
        for v1_label in LABELS_V1:
            v2_label = V1_TO_V2[v1_label]
            back_to_v1 = V2_TO_V1[v2_label]
            assert back_to_v1 == v1_label

    def test_legacy_to_v2_exists(self):
        """All legacy labels should have V2 mappings."""
        for label in LABELS_LEGACY:
            assert label in LEGACY_TO_V2

    def test_legacy_to_v1_exists(self):
        """All legacy labels should have V1 mappings."""
        for label in LABELS_LEGACY:
            assert label in LEGACY_TO_V1


class TestGetLabels:
    """Test get_labels function."""

    def test_get_v1_labels(self):
        labels = get_labels("v1")
        assert labels == LABELS_V1

    def test_get_v2_labels(self):
        labels = get_labels("v2")
        assert labels == LABELS_V2

    def test_get_legacy_labels(self):
        labels = get_labels("legacy")
        assert labels == LABELS_LEGACY

    def test_get_labels_invalid_taxonomy(self):
        with pytest.raises(ValueError, match="Unknown taxonomy"):
            get_labels("invalid")


class TestConvertLabel:
    """Test convert_label function."""

    def test_convert_v1_to_v2(self):
        assert convert_label("person_or_group", "v1", "v2") == "creator_or_studio"
        assert convert_label("content", "v1", "v2") == "content_subject"
        assert convert_label("media_bucket", "v1", "v2") == "asset_type"
        assert convert_label("descriptor", "v1", "v2") == "theme_or_genre"
        assert convert_label("other", "v1", "v2") == "other"
        assert convert_label("unknown", "v1", "v2") == "unknown"

    def test_convert_v2_to_v1(self):
        assert convert_label("creator_or_studio", "v2", "v1") == "person_or_group"
        assert convert_label("content_subject", "v2", "v1") == "content"
        assert convert_label("asset_type", "v2", "v1") == "media_bucket"
        assert convert_label("theme_or_genre", "v2", "v1") == "descriptor"
        assert convert_label("other", "v2", "v1") == "other"
        assert convert_label("unknown", "v2", "v1") == "unknown"

    def test_convert_legacy_to_v2(self):
        assert convert_label("primary_author", "legacy", "v2") == "creator_or_studio"
        assert convert_label("secondary_author", "legacy", "v2") == "creator_or_studio"
        assert convert_label("collection", "legacy", "v2") == "content_subject"
        assert convert_label("subject", "legacy", "v2") == "content_subject"
        assert convert_label("media_format", "legacy", "v2") == "asset_type"
        assert convert_label("media_type", "legacy", "v2") == "asset_type"
        assert convert_label("variant", "legacy", "v2") == "theme_or_genre"
        assert convert_label("other", "legacy", "v2") == "other"

    def test_convert_legacy_to_v1(self):
        assert convert_label("primary_author", "legacy", "v1") == "person_or_group"
        assert convert_label("secondary_author", "legacy", "v1") == "person_or_group"
        assert convert_label("collection", "legacy", "v1") == "content"
        assert convert_label("subject", "legacy", "v1") == "content"
        assert convert_label("media_format", "legacy", "v1") == "media_bucket"
        assert convert_label("media_type", "legacy", "v1") == "media_bucket"
        assert convert_label("variant", "legacy", "v1") == "descriptor"
        assert convert_label("other", "legacy", "v1") == "other"

    def test_convert_same_taxonomy(self):
        """Converting within same taxonomy should return original."""
        assert convert_label("person_or_group", "v1", "v1") == "person_or_group"
        assert convert_label("creator_or_studio", "v2", "v2") == "creator_or_studio"
        assert convert_label("primary_author", "legacy", "legacy") == "primary_author"

    def test_convert_invalid_label_returns_original(self):
        """Unknown labels should be returned as-is."""
        assert convert_label("invalid_label", "v1", "v2") == "invalid_label"

    def test_convert_invalid_taxonomy_raises(self):
        """Invalid taxonomy names should raise ValueError."""
        with pytest.raises(ValueError):
            convert_label("person_or_group", "invalid", "v2")

        with pytest.raises(ValueError):
            convert_label("person_or_group", "v1", "invalid")


class TestIsValidLabel:
    """Test is_valid_label function."""

    def test_valid_v1_labels(self):
        assert is_valid_label("person_or_group", "v1")
        assert is_valid_label("content", "v1")
        assert is_valid_label("other", "v1")

    def test_valid_v2_labels(self):
        assert is_valid_label("creator_or_studio", "v2")
        assert is_valid_label("content_subject", "v2")
        assert is_valid_label("other", "v2")

    def test_valid_legacy_labels(self):
        assert is_valid_label("primary_author", "legacy")
        assert is_valid_label("collection", "legacy")
        assert is_valid_label("other", "legacy")

    def test_invalid_labels(self):
        assert not is_valid_label("invalid_label", "v1")
        assert not is_valid_label("creator_or_studio", "v1")  # v2 label in v1
        assert not is_valid_label("person_or_group", "v2")  # v1 label in v2

    def test_invalid_taxonomy_raises(self):
        with pytest.raises(ValueError):
            is_valid_label("person_or_group", "invalid")


class TestNormalizeLabels:
    """Test normalize_labels function."""

    def test_normalize_v1_to_v2(self):
        v1_labels = {"person_or_group", "content", "media_bucket"}
        v2_labels = normalize_labels(v1_labels, "v2")
        assert v2_labels == {"creator_or_studio", "content_subject", "asset_type"}

    def test_normalize_v2_to_v1(self):
        v2_labels = {"creator_or_studio", "content_subject", "asset_type"}
        v1_labels = normalize_labels(v2_labels, "v1")
        assert v1_labels == {"person_or_group", "content", "media_bucket"}

    def test_normalize_legacy_to_v2(self):
        legacy_labels = {"primary_author", "collection", "media_type"}
        v2_labels = normalize_labels(legacy_labels, "v2")
        assert v2_labels == {"creator_or_studio", "content_subject", "asset_type"}

    def test_normalize_same_taxonomy(self):
        """Normalizing to same taxonomy should return original."""
        v2_labels = {"creator_or_studio", "content_subject"}
        result = normalize_labels(v2_labels, "v2")
        assert result == v2_labels

    def test_normalize_mixed_labels(self):
        """Mixed labels that don't match a taxonomy should return as-is."""
        mixed_labels = {"person_or_group", "creator_or_studio", "unknown_label"}
        result = normalize_labels(mixed_labels, "v2")
        # Since it doesn't match any taxonomy perfectly, return as-is
        assert result == mixed_labels


class TestVariantTypeMapping:
    """Test variant type to taxonomy mapping."""

    def test_variant_type_to_taxonomy_defined(self):
        """Ensure all common variant types have mappings."""
        assert "variant" in VARIANT_TYPE_TO_TAXONOMY
        assert "media_type" in VARIANT_TYPE_TO_TAXONOMY
        assert "media_format" in VARIANT_TYPE_TO_TAXONOMY

    def test_variant_type_mappings_complete(self):
        """Each variant type should map to both v1 and v2."""
        for variant_type, (v1_label, v2_label) in VARIANT_TYPE_TO_TAXONOMY.items():
            assert v1_label in LABELS_V1
            assert v2_label in LABELS_V2

    def test_build_variant_mappings_basic(self):
        """Test building variant mappings from config."""
        variants = {
            "winter": {"type": "variant", "synonyms": []},
            "VTT": {"type": "media_type", "synonyms": []},
            "PDF": {"type": "media_format", "synonyms": ["PDFs"]},
        }

        v1_map, v2_map = build_variant_mappings(variants)

        # Check v1 mappings
        assert v1_map["winter"] == "descriptor"
        assert v1_map["VTT"] == "media_bucket"
        assert v1_map["PDF"] == "media_bucket"

        # Check v2 mappings
        assert v2_map["winter"] == "theme_or_genre"
        assert v2_map["VTT"] == "asset_type"
        assert v2_map["PDF"] == "asset_type"

    def test_build_variant_mappings_synonyms(self):
        """Test that synonyms are also mapped."""
        variants = {
            "PDF": {"type": "media_format", "synonyms": ["PDFs", "pdf"]},
        }

        v1_map, v2_map = build_variant_mappings(variants)

        # Main name
        assert v1_map["PDF"] == "media_bucket"
        assert v2_map["PDF"] == "asset_type"

        # Synonyms
        assert v1_map["PDFs"] == "media_bucket"
        assert v1_map["pdf"] == "media_bucket"
        assert v2_map["PDFs"] == "asset_type"
        assert v2_map["pdf"] == "asset_type"

    def test_build_variant_mappings_default_type(self):
        """Test that variants without explicit type default to 'variant'."""
        variants = {
            "winter": {"synonyms": []},  # No type specified
        }

        v1_map, v2_map = build_variant_mappings(variants)

        # Should default to variant type
        assert v1_map["winter"] == "descriptor"
        assert v2_map["winter"] == "theme_or_genre"

    def test_build_variant_mappings_unknown_type(self):
        """Test that unknown types are ignored."""
        variants = {
            "unknown_thing": {"type": "unknown_type", "synonyms": []},
        }

        v1_map, v2_map = build_variant_mappings(variants)

        # Should not be in the mapping
        assert "unknown_thing" not in v1_map
        assert "unknown_thing" not in v2_map

    def test_build_variant_mappings_empty(self):
        """Test with empty variants dict."""
        variants = {}
        v1_map, v2_map = build_variant_mappings(variants)
        assert v1_map == {}
        assert v2_map == {}
