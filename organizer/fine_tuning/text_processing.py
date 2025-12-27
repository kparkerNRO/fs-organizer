"""Text processing utilities for classification.

This module provides common text normalization, tokenization, and similarity
functions used across the classification pipeline.
"""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Set, Tuple


# Regex patterns
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


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


def tokenize_string(s: str) -> List[str]:
    """Extract alphanumeric tokens from a string.

    Args:
        s: Input string

    Returns:
        List of lowercase alphanumeric tokens
    """
    return TOKEN_RE.findall(normalize_string(s))


def text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using SequenceMatcher.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0 and 1, where 1 is identical
    """
    return SequenceMatcher(None, normalize_string(text1), normalize_string(text2)).ratio()


def has_close_text_match(
    target: str,
    candidates: List[str],
    threshold: float = 0.85
) -> Tuple[bool, List[str]]:
    """Check if target has a close text match in candidates.

    Matches using exact match, substring match, or fuzzy similarity.

    Args:
        target: Text to match
        candidates: List of candidate strings
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

    Matches using exact match or whole-word containment (word boundaries).

    Args:
        text: Text to check
        patterns: List of patterns to match against

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
        if ' ' not in normalized_pattern:
            if normalized_pattern in words:
                matches.append(pattern)
        # For multi-word patterns, use substring match
        elif normalized_pattern in normalized_text:
            matches.append(pattern)

    return len(matches) > 0, matches


def char_trigrams(s: str) -> Set[str]:
    """Generate character trigrams from a string.

    Used for similarity-based clustering and matching.

    Args:
        s: Input string

    Returns:
        Set of character trigrams
    """
    s = re.sub(r"\s+", " ", s.strip())
    if len(s) < 3:
        return {s} if s else set()
    return {s[i : i + 3] for i in range(len(s) - 2)}


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
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
