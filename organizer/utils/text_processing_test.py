"""Tests for text processing utilities."""

from utils.text_processing import (
    char_trigrams,
    get_close_text_matches,
    get_matching_patterns,
    jaccard_similarity,
    normalize_string,
    tokenize_string,
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


class TestHasCloseTextMatch:
    """Test has_close_text_match function."""

    def test_exact_match(self):
        candidates = ["winter", "summer", "spring"]
        matches = get_close_text_matches("Winter", candidates)
        assert len(matches) > 0
        assert "winter" in matches

    # def test_substring_match(self):
    #     candidates = ["VTT Maps", "Print Maps"]
    #     matches = get_close_text_matches("VTT", candidates)
    #     assert len(matches) > 0
    #     assert "VTT Maps" in matches

    def test_fuzzy_match(self):
        candidates = ["Gridded", "Grid"]
        matches = get_close_text_matches("Gridded", candidates, threshold=0.8)
        assert len(matches) > 0

    def test_no_match(self):
        candidates = ["winter", "summer"]
        matches = get_close_text_matches("xyz", candidates)
        assert len(matches) == 0

    def test_threshold(self):
        candidates = ["hello"]
        # With high threshold, "hallo" won't match "hello"
        matches = get_close_text_matches("hallo", candidates, threshold=0.95)
        assert len(matches) == 0

        # With lower threshold, it will match
        matches = get_close_text_matches("hallo", candidates, threshold=0.7)
        assert len(matches) > 0


class TestHasPatternMatch:
    """Test has_pattern_match function."""

    def test_exact_match(self):
        patterns = ["maps", "tokens", "assets"]
        matches = get_matching_patterns("Maps", patterns)
        assert len(matches) > 0
        assert "maps" in matches

    def test_whole_word_matching(self):
        # Test whole-word matching (not substring)
        patterns = ["maps", "token"]
        matches = get_matching_patterns("VTT Maps Pack", patterns)
        assert len(matches) > 0
        assert "maps" in matches

    def test_partial_word_no_match(self):
        # Test that partial word doesn't match
        patterns = ["map", "pack"]
        matches = get_matching_patterns("VTT Maps Pack", patterns)
        # "map" won't match "maps" (different words)
        # but "pack" will match "Pack"
        assert len(matches) > 0
        assert "pack" in matches
        assert "map" not in matches

    def test_no_match(self):
        patterns = ["maps", "tokens"]
        matches = get_matching_patterns("xyz", patterns)
        assert len(matches) == 0

    def test_multi_word_pattern(self):
        # Multi-word patterns use substring matching
        patterns = ["VTT Maps"]
        matches = get_matching_patterns("VTT Maps Pack", patterns)
        assert len(matches) > 0
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
