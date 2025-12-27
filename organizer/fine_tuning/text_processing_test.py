"""Tests for text processing utilities."""

from fine_tuning.text_processing import (
    normalize_string,
    text_similarity,
    has_close_text_match,
    has_pattern_match,
    tokenize_string,
    char_trigrams,
    jaccard_similarity,
)


class TestNormalizeString:
    """Test normalize_string function."""

    def test_normalize_basic(self):
        assert normalize_string("Hello  World") == "hello world"
        assert normalize_string("  UPPER  ") == "upper"
        assert normalize_string("Multiple   Spaces") == "multiple spaces"

    def test_normalize_unicode(self):
        # NFKC normalization should handle various unicode forms
        assert normalize_string("café") == "café"
        assert normalize_string("CAFÉ") == "café"

    def test_normalize_empty(self):
        assert normalize_string("") == ""
        assert normalize_string("   ") == ""


class TestTextSimilarity:
    """Test text_similarity function."""

    def test_identical_strings(self):
        assert text_similarity("hello", "hello") == 1.0

    def test_case_insensitive(self):
        similarity = text_similarity("winter", "Winter")
        assert similarity == 1.0  # Case doesn't matter after normalization

    def test_different_strings(self):
        similarity = text_similarity("hello", "world")
        assert similarity < 0.5

    def test_similar_strings(self):
        similarity = text_similarity("hello", "hallo")
        assert 0.5 < similarity < 1.0


class TestHasCloseTextMatch:
    """Test has_close_text_match function."""

    def test_exact_match(self):
        candidates = ["winter", "summer", "spring"]
        has_match, matches = has_close_text_match("Winter", candidates)
        assert has_match
        assert "winter" in matches

    def test_substring_match(self):
        candidates = ["VTT Maps", "Print Maps"]
        has_match, matches = has_close_text_match("VTT", candidates)
        assert has_match
        assert "VTT Maps" in matches

    def test_fuzzy_match(self):
        candidates = ["Gridded", "Grid"]
        has_match, matches = has_close_text_match("Gridded", candidates, threshold=0.8)
        assert has_match

    def test_no_match(self):
        candidates = ["winter", "summer"]
        has_match, matches = has_close_text_match("xyz", candidates)
        assert not has_match
        assert len(matches) == 0

    def test_threshold(self):
        candidates = ["hello"]
        # With high threshold, "hallo" won't match "hello"
        has_match, matches = has_close_text_match("hallo", candidates, threshold=0.95)
        assert not has_match

        # With lower threshold, it will match
        has_match, matches = has_close_text_match("hallo", candidates, threshold=0.7)
        assert has_match


class TestHasPatternMatch:
    """Test has_pattern_match function."""

    def test_exact_match(self):
        patterns = ["maps", "tokens", "assets"]
        has_match, matches = has_pattern_match("Maps", patterns)
        assert has_match
        assert "maps" in matches

    def test_whole_word_matching(self):
        # Test whole-word matching (not substring)
        patterns = ["maps", "token"]
        has_match, matches = has_pattern_match("VTT Maps Pack", patterns)
        assert has_match
        assert "maps" in matches

    def test_partial_word_no_match(self):
        # Test that partial word doesn't match
        patterns = ["map", "pack"]
        has_match, matches = has_pattern_match("VTT Maps Pack", patterns)
        # "map" won't match "maps" (different words)
        # but "pack" will match "Pack"
        assert has_match
        assert "pack" in matches
        assert "map" not in matches

    def test_no_match(self):
        patterns = ["maps", "tokens"]
        has_match, matches = has_pattern_match("xyz", patterns)
        assert not has_match

    def test_multi_word_pattern(self):
        # Multi-word patterns use substring matching
        patterns = ["VTT Maps"]
        has_match, matches = has_pattern_match("VTT Maps Pack", patterns)
        assert has_match
        assert "VTT Maps" in matches


class TestTokenizeString:
    """Test tokenize_string function."""

    def test_basic_tokenization(self):
        tokens = tokenize_string("Hello World")
        assert tokens == ["hello", "world"]

    def test_alphanumeric_only(self):
        tokens = tokenize_string("Hello-World_123")
        assert tokens == ["hello", "world", "123"]

    def test_numbers(self):
        tokens = tokenize_string("Map 2023 Winter")
        assert tokens == ["map", "2023", "winter"]

    def test_empty_string(self):
        tokens = tokenize_string("")
        assert tokens == []

    def test_special_characters(self):
        tokens = tokenize_string("Hello!@#$%World")
        assert tokens == ["hello", "world"]


class TestCharTrigrams:
    """Test char_trigrams function."""

    def test_basic_trigrams(self):
        trigrams = char_trigrams("hello")
        assert trigrams == {"hel", "ell", "llo"}

    def test_short_strings(self):
        # String shorter than 3 characters
        trigrams = char_trigrams("hi")
        assert trigrams == {"hi"}

        trigrams = char_trigrams("a")
        assert trigrams == {"a"}

    def test_empty_string(self):
        trigrams = char_trigrams("")
        assert trigrams == set()

    def test_with_spaces(self):
        trigrams = char_trigrams("a b")
        assert trigrams == {"a b"}

    def test_longer_string(self):
        trigrams = char_trigrams("test")
        assert trigrams == {"tes", "est"}


class TestJaccardSimilarity:
    """Test jaccard_similarity function."""

    def test_identical_sets(self):
        a = {"hello", "world"}
        b = {"hello", "world"}
        assert jaccard_similarity(a, b) == 1.0

    def test_disjoint_sets(self):
        a = {"hello"}
        b = {"world"}
        assert jaccard_similarity(a, b) == 0.0

    def test_partial_overlap(self):
        a = {"hello", "world", "foo"}
        b = {"hello", "world", "bar"}
        # Intersection: {hello, world} = 2
        # Union: {hello, world, foo, bar} = 4
        # Similarity: 2/4 = 0.5
        assert jaccard_similarity(a, b) == 0.5

    def test_empty_sets(self):
        a = set()
        b = set()
        assert jaccard_similarity(a, b) == 1.0  # Both empty = identical

    def test_one_empty_set(self):
        a = {"hello"}
        b = set()
        assert jaccard_similarity(a, b) == 0.0

    def test_subset(self):
        a = {"hello", "world"}
        b = {"hello", "world", "foo"}
        # Intersection: {hello, world} = 2
        # Union: {hello, world, foo} = 3
        # Similarity: 2/3
        assert abs(jaccard_similarity(a, b) - 2 / 3) < 0.001
