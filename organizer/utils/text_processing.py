"""Text processing utilities for classification.

This module provides common text normalization, tokenization, and similarity
functions used across the classification pipeline.
"""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Set

# Regex patterns
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

###################################
# String processing
###################################


def normalize_string(s: str) -> str:
    """Normalize a string by lowercasing and removing extra whitespace.

    Uses NFKC normalization for consistent unicode handling.

    Args:
        s: Input string

    Returns:
        Normalized string
    """
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize_string(s: str) -> list[str]:
    """Extract alphanumeric tokens from a string.

    Args:
        s: Input string

    Returns:
        list of lowercase alphanumeric tokens
    """
    return TOKEN_RE.findall(normalize_string(s))


def char_trigrams(s: str) -> set[str]:
    """Generate character trigrams from a string.

    Used for similarity-based clustering and matching.

    Args:
        s: Input string

    Returns:
        set of character trigrams
    """
    s = re.sub(r"\s+", " ", s.strip())
    if len(s) < 3:
        return {s} if s else set()
    return {s[i : i + 3] for i in range(len(s) - 2)}


###################################
# Similarity checks
###################################


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Calculate Jaccard similarity between two sets.

    Args:
        a: First set
        b: Second set

    Returns:
        Jaccard similarity (0-1)
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def get_close_text_matches(
    target: str, candidates: list[str], threshold: float = 0.85
) -> list[str]:
    """Check if target has a close text match in candidates.

    Matches using exact match, or fuzzy similarity.

    Args:
        target: Text to match
        candidates: list of candidate strings
        threshold: Similarity threshold (0-1) for fuzzy matching

    Returns:
        Tuple of (has_match, list_of_matches)
    """
    matches = []
    normalized_target = normalize_string(target)

    for candidate in candidates:
        normalized_candidate = normalize_string(candidate)

        # Exact match
        if normalized_target == normalized_candidate:
            matches.append(candidate)
            continue

        if normalized_target in normalized_candidate:
            matches.append(candidate)
            continue

        # Fuzzy match
        similarity = SequenceMatcher(
            None, normalize_string(target), normalize_string(candidate)
        ).ratio()
        if similarity >= threshold:
            matches.append(candidate)

    return matches


def has_close_match(s: str, candidates: list[str], threshold: float = 0.85) -> bool:
    """
    Checks to see if a string is a close match to any in a list of strings

    Returns:
        True if any candidate has distance <= max_distance, False otherwise
    """
    return len(get_close_text_matches(s, candidates, threshold)) > 0


def get_matching_patterns(text: str, patterns: list[str]) -> list[str]:
    """Check if text matches any of the given patterns (case-insensitive).

    Matches using exact match or whole-word containment (word boundaries).

    Args:
        text: Text to check
        patterns: list of patterns to match against

    Returns:
        Tuple of (has_match, list_of_matching_patterns)
    """
    matches = []
    normalized_text = normalize_string(text)
    # Create word set for faster word-boundary matching
    words = set(normalized_text.split())

    for pattern in patterns:
        normalized_pattern = normalize_string(pattern)

        # Exact match
        if normalized_text == normalized_pattern:
            matches.append(pattern)
            continue

        # Word boundary match - check if pattern appears as a complete word or phrase
        # For single-word patterns, check word set
        if " " not in normalized_pattern:
            if normalized_pattern in words:
                matches.append(pattern)
        # For multi-word patterns, use substring match
        elif normalized_pattern in normalized_text:
            matches.append(pattern)

    return matches


def has_matching_token(token_list: List[str], cue_set: Set[str]) -> bool:
    return any(has_close_match(t, list(cue_set)) for t in token_list)
