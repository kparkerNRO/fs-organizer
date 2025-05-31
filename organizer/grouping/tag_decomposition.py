"""
Tag Decomposition Module

This module implements N-gram Component Analysis and Embedding Compositionality detection
to identify compound tags that should be decomposed into multiple component tags.

Based on analysis of real data, this focuses on:
1. Multi-word compound tags (e.g., "Castle Tower" -> "Castle", "Tower")
2. Semantic decomposition using frequency analysis of components
3. Creating derived tags without requiring standalone tag existence

The analysis operates on GroupCategoryEntry records and creates new entries for detected
component tags, maintaining confidence scores and derivation tracking.
"""

import logging
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy import select
import networkx as nx

from data_models.database import GroupCategoryEntry

# Configure logger to write to stdout
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Configuration constants
MIN_COMPOUND_LENGTH = 2  # Minimum words for compound analysis
MIN_COMPONENT_FREQUENCY = (
    2  # Minimum frequency for component to be considered significant
)
MIN_DECOMPOSITION_CONFIDENCE = 0.4  # Minimum confidence to create decomposed tags
SEMANTIC_SIMILARITY_THRESHOLD = 0.3  # Threshold for semantic similarity

# Title case configuration
STOP_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
}

COMMON_ACRONYMS = {
    "ai",
    "api",
    "ar",
    "vr",
    "3d",
    "2d",
    "hd",
    "4k",
    "8k",
    "ui",
    "ux",
    "pc",
    "npc",
    "rpg",
    "mmo",
    "fps",
    "rts",
    "dnd",
    "d&d",
    "pdf",
    "jpg",
    "png",
    "gif",
    "mp3",
    "mp4",
    "usb",
    "cd",
    "dvd",
    "tv",
    "gps",
    "wifi",
    "cpu",
    "gpu",
    "ram",
    "ssd",
    "hdd",
    "os",
    "ios",
    "app",
    "url",
    "http",
    "https",
    "ftp",
    "ssh",
    "sql",
    "html",
    "css",
    "js",
    "xml",
    "json",
    "csv",
    "exe",
    "zip",
    "rar",
    "tar",
    "gz",
}
HIGH_FREQUENCY_THRESHOLD = (
    3  # Words appearing this many times are considered significant
)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
MIN_HIERARCHICAL_GROUP_SIZE = 2  # Minimum tags needed for hierarchical analysis


def format_tag_title_case(tag: str) -> str:
    """
    Format a tag using proper title case with special handling for:
    - Acronyms (keep uppercase)
    - Stop words (keep lowercase unless first/last word)
    - Numbers with units (e.g., "3d", "4k")
    - Special characters and punctuation
    """
    if not tag or not tag.strip():
        return tag

    # Handle special cases like "d&d"
    if tag.lower() in COMMON_ACRONYMS:
        if "&" in tag:
            return tag.upper()
        return tag.upper()

    words = tag.split()
    if not words:
        return tag

    formatted_words = []

    for i, word in enumerate(words):
        # Clean word of punctuation at start/end for checking
        clean_word = word.strip(".,!?;:()[]{}\"'").lower()

        # Always capitalize first and last word
        if i == 0 or i == len(words) - 1:
            if clean_word in COMMON_ACRONYMS:
                formatted_words.append(word.upper())
            else:
                formatted_words.append(word.capitalize())

        # Handle acronyms
        elif clean_word in COMMON_ACRONYMS:
            formatted_words.append(word.upper())

        # Handle stop words (keep lowercase)
        elif clean_word in STOP_WORDS:
            formatted_words.append(word.lower())

        # Handle numbers with letters (like "3d", "4k")
        elif clean_word.isalnum() and any(c.isdigit() for c in clean_word):
            formatted_words.append(word.upper())

        # Regular words - capitalize
        else:
            formatted_words.append(word.capitalize())

    return " ".join(formatted_words)


@dataclass
class DecompositionCandidate:
    """Represents a potential tag decomposition with confidence metrics"""

    original_tag: str
    components: List[str]
    frequency_score: float
    semantic_score: float
    structural_score: float
    embedding_score: float
    cooccurrence_score: float
    pattern_score: float
    overall_confidence: float
    evidence_types: List[str]
    folder_id: int
    original_entry: GroupCategoryEntry


@dataclass
class ComponentStatistics:
    """Statistics for component words and their relationships"""

    word_frequencies: Dict[str, int]
    word_contexts: Dict[str, Set[str]]  # Words that appear with this word
    compound_patterns: Dict[str, List[str]]  # Common patterns for each word
    cooccurrence_matrix: Dict[str, Dict[str, int]]  # Tag co-occurrence data
    ngram_frequencies: Dict[str, int]  # N-gram component frequencies
    prefix_patterns: Dict[str, List[str]]  # Common prefix patterns
    suffix_patterns: Dict[str, List[str]]  # Common suffix patterns
    total_words: int
    total_compounds: int


class TagDecomposer:
    """Main class for analyzing and decomposing compound tags using multi-modal analysis"""

    def __init__(self, min_confidence: float = MIN_DECOMPOSITION_CONFIDENCE):
        self.min_confidence = min_confidence
        self.min_component_frequency = MIN_COMPONENT_FREQUENCY

        # Initialize ML models
        self.vectorizer = TfidfVectorizer(
            stop_words=None,
            ngram_range=(1, 2),  # Include both unigrams and bigrams
            min_df=1,
            max_features=1000,
        )

        # Initialize sentence transformer for embeddings
        logger.info(f"Loading sentence transformer model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # Data storage
        self.component_stats = None
        self.tag_vectors = None
        self.embeddings = None
        self.vocabulary = []
        self.tag_to_idx = {}
        self.original_id_to_normalized = {}  # Mapping from entry IDs to normalized versions
        self.normalized_to_original_id = {}  # Mapping from normalized tags to list of entry IDs

        # Analysis results
        self.decomposition_candidates = {}
        self.component_graph = nx.Graph()
        self.atomic_components = set()

    def extract_component_statistics(
        self, entries: List[GroupCategoryEntry]
    ) -> ComponentStatistics:
        """Extract comprehensive statistical information about word components in compound tags"""
        logger.info(f"Extracting component statistics from {len(entries)} entries")

        word_frequencies = Counter()
        word_contexts = defaultdict(set)
        compound_patterns = defaultdict(list)
        cooccurrence_matrix = defaultdict(lambda: defaultdict(int))
        ngram_frequencies = Counter()
        prefix_patterns = defaultdict(list)
        suffix_patterns = defaultdict(list)
        total_words = 0
        total_compounds = 0

        # Build vocabulary for embedding generation
        clean_tags = []

        for entry in entries:
            if not entry.processed_name:
                continue

            # Normalize the tag to handle variations like / vs spaces
            original_tag = entry.processed_name.strip()
            normalized_tag = original_tag.lower()

            if not normalized_tag or len(normalized_tag) <= 1:
                continue

            # Build mapping between entry IDs and normalized tags
            self.original_id_to_normalized[entry.id] = normalized_tag
            if normalized_tag not in self.normalized_to_original_id:
                self.normalized_to_original_id[normalized_tag] = []
            self.normalized_to_original_id[normalized_tag].append(entry.id)

            clean_tags.append(normalized_tag)
            words = normalized_tag.split()

            # Track all tags for vocabulary (use normalized)
            if len(words) >= 1:
                self.vocabulary.append(normalized_tag)

            # Skip single words for compound analysis
            if len(words) < 2:
                continue

            total_compounds += 1

            # Extract n-grams (1-4 words)
            for n in range(1, min(5, len(words) + 1)):
                for i in range(len(words) - n + 1):
                    ngram = " ".join(words[i : i + n])
                    if len(ngram.strip()) > 1:
                        ngram_frequencies[ngram] += 1

            # Track word frequencies and contexts
            for i, word in enumerate(words):
                if len(word) > 1:  # Skip single characters
                    word_frequencies[word] += 1
                    total_words += 1

                    # Track context (words that appear with this word)
                    context_words = [
                        w for j, w in enumerate(words) if i != j and len(w) > 1
                    ]
                    word_contexts[word].update(context_words)

                    # Track compound patterns
                    compound_patterns[word].append(normalized_tag)

            # Build co-occurrence matrix (using normalized tags)
            for other_entry in entries:
                if other_entry.processed_name and other_entry != entry:
                    other_normalized = other_entry.processed_name.strip().lower()
                    if other_normalized and other_normalized != normalized_tag:
                        # Simple co-occurrence based on shared words
                        tag_words = set(normalized_tag.split())
                        other_words = set(other_normalized.split())
                        if tag_words & other_words:  # If they share words
                            cooccurrence_matrix[normalized_tag][other_normalized] += 1

            # Extract prefix/suffix patterns
            if len(words) >= 2:
                # Prefix patterns (first 1-2 words)
                for prefix_len in range(1, min(3, len(words))):
                    prefix = " ".join(words[:prefix_len])
                    suffix = " ".join(words[prefix_len:])
                    if len(suffix.split()) >= 1:
                        prefix_patterns[prefix].append(normalized_tag)

                # Suffix patterns (last 1-2 words)
                for suffix_len in range(1, min(3, len(words))):
                    suffix = " ".join(words[-suffix_len:])
                    prefix = " ".join(words[:-suffix_len])
                    if len(prefix.split()) >= 1:
                        suffix_patterns[suffix].append(normalized_tag)

        # Remove duplicates from vocabulary and build tag_to_idx mapping
        self.vocabulary = list(set(self.vocabulary))
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.vocabulary)}

        logger.info(f"Built vocabulary of {len(self.vocabulary)} unique tags")
        logger.info(f"Found {len(ngram_frequencies)} unique n-grams")

        return ComponentStatistics(
            word_frequencies=dict(word_frequencies),
            word_contexts={k: v for k, v in word_contexts.items()},
            compound_patterns={k: v for k, v in compound_patterns.items()},
            cooccurrence_matrix=dict(cooccurrence_matrix),
            ngram_frequencies=dict(ngram_frequencies),
            prefix_patterns=dict(prefix_patterns),
            suffix_patterns=dict(suffix_patterns),
            total_words=total_words,
            total_compounds=total_compounds,
        )

    def compute_tag_vectors_and_embeddings(
        self,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Compute both TF-IDF vectors and semantic embeddings for tags"""
        logger.info("Computing TF-IDF vectors and semantic embeddings")

        if not self.vocabulary:
            logger.warning("No vocabulary available for vector computation")
            return {}, np.array([])

        # Compute TF-IDF vectors
        try:
            vectors_matrix = self.vectorizer.fit_transform(self.vocabulary)
            tag_vectors = {}
            for i, tag in enumerate(self.vocabulary):
                tag_vectors[tag] = vectors_matrix[i].toarray().flatten()
        except Exception as e:
            logger.warning(f"Failed to compute TF-IDF vectors: {e}")
            tag_vectors = {}

        # Compute semantic embeddings
        try:
            logger.info("Generating semantic embeddings...")
            embeddings = self.embedding_model.encode(self.vocabulary)
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        except Exception as e:
            logger.warning(f"Failed to compute embeddings: {e}")
            embeddings = np.array([])

        return tag_vectors, embeddings

    def calculate_frequency_score(self, components: List[str]) -> float:
        """Calculate score based on component n-gram frequencies"""
        if not self.component_stats:
            return 0.0

        scores = []
        for component in components:
            component_lower = component.lower()

            # Check both word frequency and n-gram frequency
            word_freq = self.component_stats.word_frequencies.get(component_lower, 0)
            ngram_freq = self.component_stats.ngram_frequencies.get(component_lower, 0)

            # Use the higher of the two frequencies
            frequency = max(word_freq, ngram_freq)

            if frequency >= HIGH_FREQUENCY_THRESHOLD:
                score = min(1.0, frequency / 20.0)
            elif frequency >= self.min_component_frequency:
                score = 0.5 + (frequency / 10.0)
            else:
                score = frequency / 5.0

            scores.append(min(score, 1.0))

        # Return geometric mean of component scores
        if scores:
            return np.exp(np.mean(np.log(np.array(scores) + 1e-8)))
        return 0.0

    def calculate_semantic_score(
        self, original_tag: str, components: List[str]
    ) -> float:
        """Calculate semantic coherence score using word contexts"""
        if not self.component_stats or len(components) < 2:
            return 0.0

        # Check if components often appear together in contexts
        context_overlap_scores = []

        for i, comp1 in enumerate(components):
            comp1_lower = comp1.lower()
            comp1_contexts = self.component_stats.word_contexts.get(comp1_lower, set())

            for j, comp2 in enumerate(components):
                if i >= j:  # Avoid double counting
                    continue

                comp2_lower = comp2.lower()
                comp2_contexts = self.component_stats.word_contexts.get(
                    comp2_lower, set()
                )

                # Calculate Jaccard similarity of contexts
                if comp1_contexts and comp2_contexts:
                    intersection = len(comp1_contexts & comp2_contexts)
                    union = len(comp1_contexts | comp2_contexts)
                    jaccard = intersection / union if union > 0 else 0.0
                    context_overlap_scores.append(jaccard)

                # Bonus if they appear in each other's contexts
                if comp2_lower in comp1_contexts or comp1_lower in comp2_contexts:
                    context_overlap_scores.append(0.5)

        return np.mean(context_overlap_scores) if context_overlap_scores else 0.0

    def extract_ngram_components(
        self, max_n: int = 4
    ) -> Tuple[Dict[str, List[Tuple[List[str], float]]], Dict[str, int]]:
        """Extract potential components using n-gram analysis"""
        logger.info("Extracting n-gram components...")

        if not self.component_stats:
            return {}, {}

        # Filter n-grams by frequency
        frequent_ngrams = {
            ngram: count
            for ngram, count in self.component_stats.ngram_frequencies.items()
            if count >= self.min_component_frequency and len(ngram.split()) <= 3
        }

        logger.info(f"Found {len(frequent_ngrams)} frequent n-gram components")

        # Find tags that can be decomposed using these n-grams
        decomposition_candidates = {}

        for tag in self.vocabulary:
            if len(tag.split()) > 1:  # Only consider multi-word tags
                potential_components = self._find_component_combinations(
                    tag, frequent_ngrams
                )
                if potential_components:
                    decomposition_candidates[tag] = potential_components

        return decomposition_candidates, frequent_ngrams

    def _find_component_combinations(
        self, tag: str, components: Dict[str, int]
    ) -> List[Tuple[List[str], float]]:
        """Find all ways to decompose a tag using available components"""
        words = tag.split()
        tag_length = len(words)

        valid_decompositions = []

        def find_decompositions(start_idx: int, current_components: List[str]):
            if start_idx == tag_length:
                if len(current_components) >= 2:  # Must have at least 2 components
                    valid_decompositions.append(current_components.copy())
                return

            # Try components of different lengths starting from current position
            for length in range(1, tag_length - start_idx + 1):
                candidate = " ".join(words[start_idx : start_idx + length])

                if candidate in components:
                    current_components.append(candidate)
                    find_decompositions(start_idx + length, current_components)
                    current_components.pop()

        find_decompositions(0, [])

        # Score decompositions by component frequency and coverage
        scored_decompositions = []
        for decomp in valid_decompositions:
            score = self._score_decomposition(tag, decomp, components)
            scored_decompositions.append((decomp, score))

        # Return best decompositions
        scored_decompositions.sort(key=lambda x: x[1], reverse=True)
        return scored_decompositions[:3]  # Top 3 decompositions

    def _score_decomposition(
        self,
        original_tag: str,
        components: List[str],
        component_frequencies: Dict[str, int],
    ) -> float:
        """Score how good a decomposition is"""
        # Base score from component frequencies (more frequent = better)
        freq_score = np.mean([component_frequencies[comp] for comp in components])
        freq_score = min(freq_score / 10, 1.0)  # Normalize

        # Penalty for too many tiny components
        length_penalty = 1.0 if len(components) <= 3 else 0.8

        # Bonus for components that exist as standalone tags
        standalone_bonus = sum(
            1 for comp in components if comp in self.tag_to_idx
        ) / len(components)

        # Penalty for leaving single character components
        char_penalty = 1.0 if all(len(comp) > 1 for comp in components) else 0.7

        total_score = (
            freq_score * length_penalty * (1 + standalone_bonus * 0.5) * char_penalty
        )

        return total_score

    def calculate_structural_score(
        self, original_tag: str, components: List[str]
    ) -> float:
        """Calculate structural decomposition score based on tag structure"""
        words = original_tag.lower().split()
        component_words = [comp.lower() for comp in components]

        # Check coverage - do components cover the original tag?
        covered_words = set()
        for component in component_words:
            comp_words = component.split()
            covered_words.update(comp_words)

        original_words = set(words)
        coverage = (
            len(covered_words & original_words) / len(original_words)
            if original_words
            else 0.0
        )

        # Bonus for perfect coverage
        perfect_coverage_bonus = 0.3 if coverage >= 0.8 else 0.0

        # Penalty for too many components relative to original words
        component_penalty = max(0.0, 0.1 * (len(components) - len(words)))

        return max(0.0, coverage + perfect_coverage_bonus - component_penalty)

    def calculate_embedding_score(
        self, original_tag: str, components: List[str]
    ) -> float:
        """Calculate embedding-based compositionality score"""
        if self.embeddings is None or original_tag not in self.tag_to_idx:
            return 0.0

        try:
            # Get embedding for original tag
            tag_idx = self.tag_to_idx[original_tag]
            original_embedding = self.embeddings[tag_idx]

            # Get embeddings for components
            component_embeddings = []
            for component in components:
                if component in self.tag_to_idx:
                    comp_idx = self.tag_to_idx[component]
                    component_embeddings.append(self.embeddings[comp_idx])
                else:
                    # Generate embedding for unknown component
                    comp_emb = self.embedding_model.encode([component])[0]
                    component_embeddings.append(comp_emb)

            if len(component_embeddings) >= 2:
                # Calculate compositional embedding (average of components)
                compositional_embedding = np.mean(component_embeddings, axis=0)

                # Calculate cosine similarity
                similarity = cosine_similarity(
                    [original_embedding], [compositional_embedding]
                )[0][0]

                return max(0.0, similarity)

        except Exception as e:
            logger.debug(
                f"Failed to calculate embedding score for '{original_tag}': {e}"
            )

        return 0.0

    def calculate_cooccurrence_score(
        self, original_tag: str, components: List[str]
    ) -> float:
        """Calculate co-occurrence based score"""
        if not self.component_stats or not self.component_stats.cooccurrence_matrix:
            return 0.0

        tag_coocs = self.component_stats.cooccurrence_matrix.get(original_tag, {})
        if not tag_coocs:
            return 0.0

        scores = []
        for component in components:
            if component in self.component_stats.cooccurrence_matrix:
                comp_coocs = self.component_stats.cooccurrence_matrix[component]

                # Calculate Jaccard similarity of co-occurrence partners
                tag_partners = set(tag_coocs.keys())
                comp_partners = set(comp_coocs.keys())

                if tag_partners and comp_partners:
                    overlap = len(tag_partners & comp_partners)
                    union = len(tag_partners | comp_partners)
                    jaccard = overlap / union if union > 0 else 0.0
                    scores.append(jaccard)

        return np.mean(scores) if scores else 0.0

    def calculate_pattern_score(
        self, original_tag: str, components: List[str]
    ) -> float:
        """Calculate pattern-based score (prefix/suffix analysis)"""
        if not self.component_stats:
            return 0.0

        words = original_tag.split()
        if len(words) < 2:
            return 0.0

        score = 0.0

        # Check if components match common prefix patterns
        for i in range(1, len(words)):
            prefix = " ".join(words[:i]).lower()
            if prefix in self.component_stats.prefix_patterns:
                pattern_frequency = len(self.component_stats.prefix_patterns[prefix])
                if (
                    prefix in [comp.lower() for comp in components]
                    and pattern_frequency >= 2
                ):
                    score += 0.5 * min(1.0, pattern_frequency / 10.0)

        # Check if components match common suffix patterns
        for i in range(1, len(words)):
            suffix = " ".join(words[-i:]).lower()
            if suffix in self.component_stats.suffix_patterns:
                pattern_frequency = len(self.component_stats.suffix_patterns[suffix])
                if (
                    suffix in [comp.lower() for comp in components]
                    and pattern_frequency >= 2
                ):
                    score += 0.5 * min(1.0, pattern_frequency / 10.0)

        return min(1.0, score)

    def analyze_hierarchical_patterns(self) -> Dict[str, Dict]:
        """Detect hierarchical tag patterns like 'Base Tag', 'Base Tag Extension', 'Base Tag Extension Extra'"""
        logger.info("Analyzing hierarchical tag patterns...")

        if not self.vocabulary:
            return {}

        hierarchical_candidates = {}

        # Group tags by common prefixes
        prefix_groups = defaultdict(list)

        # Sort tags by length to process shorter ones first
        sorted_tags = sorted(self.vocabulary, key=len)

        for tag in sorted_tags:
            words = tag.split()

            # Look for existing prefixes that this tag extends
            for existing_tag in sorted_tags:
                if existing_tag == tag or len(existing_tag.split()) >= len(words):
                    continue

                existing_words = existing_tag.split()

                # Check if existing_tag is a prefix of current tag
                if (
                    len(existing_words) < len(words)
                    and words[: len(existing_words)] == existing_words
                ):
                    # This is a hierarchical extension
                    base_tag = existing_tag
                    extension = " ".join(words[len(existing_words) :])

                    if base_tag not in prefix_groups:
                        prefix_groups[base_tag] = []

                    prefix_groups[base_tag].append(
                        {
                            "extended_tag": tag,
                            "extension": extension,
                            "base_length": len(existing_words),
                            "extension_length": len(words) - len(existing_words),
                        }
                    )

        # Process hierarchical groups
        for base_tag, extensions in prefix_groups.items():
            if len(extensions) >= MIN_HIERARCHICAL_GROUP_SIZE:
                # Extract unique extensions
                unique_extensions = set()
                extended_tags = []

                for ext_info in extensions:
                    unique_extensions.add(ext_info["extension"])
                    extended_tags.append(ext_info["extended_tag"])

                # Create hierarchical decomposition candidate
                hierarchical_candidates[f"hierarchical:{base_tag}"] = {
                    "type": "hierarchical",
                    "base_component": base_tag,
                    "extensions": list(unique_extensions),
                    "extended_tags": extended_tags,
                    "frequency": len(extensions),
                    "confidence": min(
                        1.0, len(extensions) / 5.0
                    ),  # Higher confidence with more extensions
                }

                logger.debug(
                    f"Hierarchical pattern: '{base_tag}' + {list(unique_extensions)}"
                )

        logger.info(f"Found {len(hierarchical_candidates)} hierarchical patterns")
        return hierarchical_candidates

    def detect_prefix_suffix_patterns(self) -> Dict[str, Dict]:
        """Detect tags that share common prefixes or suffixes"""
        logger.info("Detecting prefix/suffix patterns...")

        if not self.component_stats:
            return {}

        significant_patterns = {}

        # Analyze prefix patterns
        for prefix, tag_list in self.component_stats.prefix_patterns.items():
            if len(tag_list) >= 2:  # At least 2 tags share this prefix
                # Extract suffixes for these tags
                suffixes = []
                for tag in tag_list:
                    words = tag.split()
                    prefix_words = prefix.split()
                    if len(words) > len(prefix_words):
                        suffix = " ".join(words[len(prefix_words) :])
                        suffixes.append(suffix)

                if self._are_meaningful_components(suffixes):
                    significant_patterns[f"prefix:{prefix}"] = {
                        "type": "prefix",
                        "component": prefix,
                        "tags": tag_list,
                        "other_components": suffixes,
                        "frequency": len(tag_list),
                    }

        # Analyze suffix patterns
        for suffix, tag_list in self.component_stats.suffix_patterns.items():
            if len(tag_list) >= 2:
                # Extract prefixes for these tags
                prefixes = []
                for tag in tag_list:
                    words = tag.split()
                    suffix_words = suffix.split()
                    if len(words) > len(suffix_words):
                        prefix = " ".join(words[: -len(suffix_words)])
                        prefixes.append(prefix)

                if self._are_meaningful_components(prefixes):
                    significant_patterns[f"suffix:{suffix}"] = {
                        "type": "suffix",
                        "component": suffix,
                        "tags": tag_list,
                        "other_components": prefixes,
                        "frequency": len(tag_list),
                    }

        logger.info(
            f"Found {len(significant_patterns)} significant prefix/suffix patterns"
        )
        return significant_patterns

    def _are_meaningful_components(self, components: List[str]) -> bool:
        """Check if a list of components are meaningful (not just random words)"""
        if not components:
            return False

        meaningful_count = 0
        for comp in components:
            if (
                comp in self.tag_to_idx
                or any(word in self.tag_to_idx for word in comp.split())
                or len(comp.split()) == 1
            ):  # Single words are often meaningful
                meaningful_count += 1

        return (
            meaningful_count / len(components) >= 0.5
        )  # At least half should be meaningful

    def analyze_embedding_compositionality(self) -> Dict[str, Dict]:
        """Use embeddings to detect compositional tags"""
        logger.info("Analyzing embedding compositionality...")

        if self.embeddings is None or self.embeddings.size == 0:
            logger.warning("No embeddings available for compositionality analysis")
            return {}

        logger.info(f"Using embeddings with shape: {self.embeddings.shape}")

        compositional_candidates = {}

        for tag in self.vocabulary:
            words = tag.split()
            if len(words) >= 2:
                # Get embedding for the full tag
                if tag not in self.tag_to_idx:
                    continue

                tag_idx = self.tag_to_idx[tag]
                full_embedding = self.embeddings[tag_idx]

                # Try different ways to split the tag
                possible_splits = self._generate_splits(words)

                best_split = None
                best_score = 0

                for split_components in possible_splits:
                    # Check if all components exist in vocabulary or are meaningful
                    component_embeddings = []

                    for component in split_components:
                        if component in self.tag_to_idx:
                            comp_idx = self.tag_to_idx[component]
                            component_embeddings.append(self.embeddings[comp_idx])
                        else:
                            # Generate embedding for unknown component
                            try:
                                comp_emb = self.embedding_model.encode([component])[0]
                                component_embeddings.append(comp_emb)
                            except Exception as e:
                                logger.debug(
                                    f"Failed to generate embedding for component '{component}': {e}"
                                )
                                continue

                    if len(component_embeddings) >= 2:
                        # Calculate compositional embedding (average of components)
                        compositional_embedding = np.mean(component_embeddings, axis=0)

                        # Calculate similarity between full tag and compositional embedding
                        similarity = cosine_similarity(
                            [full_embedding], [compositional_embedding]
                        )[0][0]

                        if similarity > best_score:
                            best_score = similarity
                            best_split = split_components

                # If similarity is high enough, consider it compositional
                if best_score >= self.min_confidence:
                    compositional_candidates[tag] = {
                        "components": best_split,
                        "compositionality_score": best_score,
                        "method": "embedding",
                    }

        logger.info(f"Found {len(compositional_candidates)} compositional candidates")
        return compositional_candidates

    def _generate_splits(self, words: List[str]) -> List[List[str]]:
        """Generate all reasonable ways to split a list of words into components"""
        if len(words) < 2:
            return []

        splits = []

        # Binary splits
        for i in range(1, len(words)):
            left = " ".join(words[:i])
            right = " ".join(words[i:])
            splits.append([left, right])

        # Three-way splits (for longer tags)
        if len(words) >= 3:
            for i in range(1, len(words) - 1):
                for j in range(i + 1, len(words)):
                    left = " ".join(words[:i])
                    middle = " ".join(words[i:j])
                    right = " ".join(words[j:])
                    splits.append([left, middle, right])

        return splits

    def analyze_cooccurrence_patterns(self) -> Dict[str, Dict]:
        """Use co-occurrence patterns to identify decomposable tags"""
        logger.info("Analyzing co-occurrence patterns...")

        if not self.component_stats or not self.component_stats.cooccurrence_matrix:
            logger.warning("No co-occurrence data available")
            return {}

        cooccurrence_candidates = {}

        for tag in self.vocabulary:
            if len(tag.split()) >= 2:
                # Get co-occurrence partners for this tag
                tag_coocs = self.component_stats.cooccurrence_matrix.get(tag, {})

                # Generate possible components
                words = tag.split()
                possible_components = []

                # Single words
                possible_components.extend(words)

                # Two-word combinations
                for i in range(len(words) - 1):
                    two_word = " ".join(words[i : i + 2])
                    possible_components.append(two_word)

                # Check if any components have similar co-occurrence patterns
                component_matches = []

                for component in possible_components:
                    if (
                        component in self.component_stats.cooccurrence_matrix
                        and component != tag
                    ):
                        comp_coocs = self.component_stats.cooccurrence_matrix[component]

                        # Calculate overlap in co-occurrence partners
                        tag_partners = set(tag_coocs.keys())
                        comp_partners = set(comp_coocs.keys())

                        if tag_partners and comp_partners:
                            overlap = len(tag_partners & comp_partners)
                            union = len(tag_partners | comp_partners)
                            jaccard = overlap / union if union > 0 else 0

                            if jaccard > 0.3:  # Significant overlap
                                component_matches.append((component, jaccard))

                if component_matches:
                    # Sort by similarity
                    component_matches.sort(key=lambda x: x[1], reverse=True)

                    cooccurrence_candidates[tag] = {
                        "similar_components": component_matches,
                        "method": "cooccurrence",
                    }

        logger.info(f"Found {len(cooccurrence_candidates)} co-occurrence candidates")
        return cooccurrence_candidates

    def comprehensive_decomposition_analysis(self) -> Dict[str, Dict]:
        """Run all decomposition methods and combine results"""
        logger.info("=== Running Comprehensive Decomposition Analysis ===")

        if not self.component_stats:
            logger.error(
                "Component statistics not available. Call extract_component_statistics first."
            )
            return {}

        # Run all analysis methods
        ngram_candidates, frequent_ngrams = self.extract_ngram_components()
        pattern_candidates = self.detect_prefix_suffix_patterns()
        hierarchical_candidates = self.analyze_hierarchical_patterns()
        embedding_candidates = self.analyze_embedding_compositionality()
        cooccurrence_candidates = self.analyze_cooccurrence_patterns()

        # Combine results with confidence scoring
        combined_candidates = {}

        # Start with n-gram candidates (most reliable)
        for tag, decompositions in ngram_candidates.items():
            combined_candidates[tag] = {
                "decompositions": decompositions,
                "evidence": ["ngram"],
                "confidence": decompositions[0][1] if decompositions else 0,
            }

        # Add embedding evidence
        for tag, info in embedding_candidates.items():
            if tag in combined_candidates:
                combined_candidates[tag]["evidence"].append("embedding")
                combined_candidates[tag]["confidence"] += (
                    info["compositionality_score"] * 0.5
                )
            else:
                combined_candidates[tag] = {
                    "decompositions": [
                        (info["components"], info["compositionality_score"])
                    ],
                    "evidence": ["embedding"],
                    "confidence": info["compositionality_score"],
                }

        # Add co-occurrence evidence
        for tag, info in cooccurrence_candidates.items():
            if tag in combined_candidates:
                combined_candidates[tag]["evidence"].append("cooccurrence")
                combined_candidates[tag]["confidence"] += 0.2

        # Add pattern evidence
        for pattern_key, info in pattern_candidates.items():
            for tag in info["tags"]:
                if tag in combined_candidates:
                    combined_candidates[tag]["evidence"].append("pattern")
                    combined_candidates[tag]["confidence"] += 0.3

        # Add hierarchical evidence (high priority)
        for pattern_key, info in hierarchical_candidates.items():
            base_tag = info["base_component"]

            # Create decomposition for each extended tag
            for extended_tag in info["extended_tags"]:
                if extended_tag in self.vocabulary:
                    # Find which extension this tag uses
                    tag_words = extended_tag.split()
                    base_words = base_tag.split()
                    extension_words = tag_words[len(base_words) :]
                    extension = " ".join(extension_words)

                    # Check if the extension itself can be further decomposed
                    components = [base_tag]

                    # Be more conservative about decomposing extensions
                    if len(extension_words) > 1:
                        # Only decompose multi-word extensions if they meet certain criteria
                        should_decompose = False

                        # Decompose if extension words are common/frequent enough
                        extension_word_frequencies = [
                            self.component_stats.word_frequencies.get(word, 0)
                            for word in extension_words
                        ]

                        # Only decompose if most words appear frequently elsewhere
                        frequent_extension_words = sum(
                            1 for freq in extension_word_frequencies if freq >= 2
                        )
                        if (
                            frequent_extension_words >= len(extension_words) * 0.7
                        ):  # 70% of words are frequent
                            should_decompose = True

                        # Special case: very common patterns like "Location Type"
                        if len(extension_words) == 2:
                            # Check if both words appear in other contexts frequently
                            word1_freq = self.component_stats.word_frequencies.get(
                                extension_words[0], 0
                            )
                            word2_freq = self.component_stats.word_frequencies.get(
                                extension_words[1], 0
                            )
                            if word1_freq >= 2 and word2_freq >= 2:
                                should_decompose = True

                        if should_decompose:
                            # Decompose into individual words
                            for word in extension_words:
                                if len(word.strip()) > 0:
                                    components.append(word.strip())
                        else:
                            # Keep as single extension
                            components.append(extension)
                    else:
                        # Single word extension
                        components.append(extension)

                    if extended_tag in combined_candidates:
                        # Update existing candidate with hierarchical evidence
                        combined_candidates[extended_tag]["evidence"].append(
                            "hierarchical"
                        )
                        combined_candidates[extended_tag]["confidence"] += (
                            0.6  # High boost for hierarchical
                        )
                        # Update decomposition if hierarchical is better
                        combined_candidates[extended_tag]["decompositions"] = [
                            (components, info["confidence"])
                        ]
                    else:
                        # Create new candidate
                        combined_candidates[extended_tag] = {
                            "decompositions": [(components, info["confidence"])],
                            "evidence": ["hierarchical"],
                            "confidence": info["confidence"],
                        }

        # Normalize confidence scores
        for tag in combined_candidates:
            evidence_count = len(combined_candidates[tag]["evidence"])
            combined_candidates[tag]["confidence"] = min(
                combined_candidates[tag]["confidence"] / evidence_count, 1.0
            )

        # Filter by minimum confidence
        high_confidence_candidates = {
            tag: info
            for tag, info in combined_candidates.items()
            if info["confidence"] >= self.min_confidence
        }

        self.decomposition_candidates = high_confidence_candidates

        logger.info("\n=== Results Summary ===")
        logger.info(
            f"Total decomposition candidates: {len(high_confidence_candidates)}"
        )
        logger.info(
            f"High confidence (>0.8): {len([c for c in high_confidence_candidates.values() if c['confidence'] > 0.8])}"
        )
        logger.info(
            f"Medium confidence (0.6-0.8): {len([c for c in high_confidence_candidates.values() if 0.6 <= c['confidence'] <= 0.8])}"
        )

        return high_confidence_candidates

    def generate_decomposition_candidates(
        self, entries: List[GroupCategoryEntry]
    ) -> List[DecompositionCandidate]:
        """Generate candidate decompositions using multi-modal analysis"""
        logger.info("Generating decomposition candidates using multi-modal analysis")

        # Extract statistics and compute vectors/embeddings first
        self.component_stats = self.extract_component_statistics(entries)
        self.tag_vectors, self.embeddings = self.compute_tag_vectors_and_embeddings()

        # Run comprehensive analysis to get candidates
        candidate_dict = self.comprehensive_decomposition_analysis()

        # Convert to DecompositionCandidate objects with database integration
        candidates = []
        entry_by_id = {}

        # Build mapping from entry ID to entry
        for entry in entries:
            if entry.processed_name:
                entry_by_id[entry.id] = entry

        # For each normalized tag that has decomposition candidates,
        # create candidates for ALL entries that map to it
        for normalized_tag, info in candidate_dict.items():
            if normalized_tag not in self.normalized_to_original_id:
                continue

            # Get all entry IDs that map to this normalized tag
            entry_ids = self.normalized_to_original_id[normalized_tag]

            for entry_id in entry_ids:
                if entry_id not in entry_by_id:
                    continue

                entry = entry_by_id[entry_id]

                best_decomposition = (
                    info["decompositions"][0] if info["decompositions"] else ([], 0.0)
                )
                components = best_decomposition[0]
                base_score = best_decomposition[1]

                if len(components) < 2:
                    continue

                # Calculate all scoring metrics (using normalized tag for calculations)
                frequency_score = self.calculate_frequency_score(components)
                semantic_score = self.calculate_semantic_score(
                    normalized_tag, components
                )
                structural_score = self.calculate_structural_score(
                    normalized_tag, components
                )
                embedding_score = self.calculate_embedding_score(
                    normalized_tag, components
                )
                cooccurrence_score = self.calculate_cooccurrence_score(
                    normalized_tag, components
                )
                pattern_score = self.calculate_pattern_score(normalized_tag, components)

                # Multi-evidence weighted confidence
                evidence_weights = {
                    "hierarchical": 0.35,  # Highest weight for hierarchical patterns
                    "ngram": 0.20,
                    "embedding": 0.15,
                    "pattern": 0.10,
                    "cooccurrence": 0.10,
                    "frequency": 0.05,
                    "semantic": 0.03,
                    "structural": 0.02,
                }

                # Calculate hierarchical score if applicable
                hierarchical_score = 0.8 if "hierarchical" in info["evidence"] else 0.0

                overall_confidence = (
                    evidence_weights.get("hierarchical", 0) * hierarchical_score
                    + evidence_weights.get("frequency", 0) * frequency_score
                    + evidence_weights.get("semantic", 0) * semantic_score
                    + evidence_weights.get("structural", 0) * structural_score
                    + evidence_weights.get("embedding", 0) * embedding_score
                    + evidence_weights.get("cooccurrence", 0) * cooccurrence_score
                    + evidence_weights.get("pattern", 0) * pattern_score
                    + evidence_weights.get("ngram", 0) * base_score
                )

                # Boost confidence if multiple evidence types agree
                evidence_boost = min(0.2, 0.05 * len(info["evidence"]))
                overall_confidence += evidence_boost
                overall_confidence = min(1.0, overall_confidence)

                if overall_confidence >= self.min_confidence:
                    candidate = DecompositionCandidate(
                        original_tag=entry.processed_name.strip(),  # Use original tag from entry
                        components=components,
                        frequency_score=frequency_score,
                        semantic_score=semantic_score,
                        structural_score=structural_score,
                        embedding_score=embedding_score,
                        cooccurrence_score=cooccurrence_score,
                        pattern_score=pattern_score
                        + hierarchical_score,  # Include hierarchical in pattern score
                        overall_confidence=overall_confidence,
                        evidence_types=info["evidence"],
                        folder_id=entry.folder_id,
                        original_entry=entry,
                    )
                    candidates.append(candidate)

                    logger.debug(
                        f"Multi-modal candidate: '{entry.processed_name.strip()}' -> {components}"
                    )
                    logger.debug(
                        f"  Confidence: {overall_confidence:.3f}, Evidence: {info['evidence']}"
                    )
                    logger.debug(
                        f"  Scores - freq: {frequency_score:.3f}, embed: {embedding_score:.3f}, "
                        f"pattern: {pattern_score:.3f}, cooc: {cooccurrence_score:.3f}"
                    )

        logger.info(f"Generated {len(candidates)} multi-modal decomposition candidates")
        return candidates

    def create_decomposed_entries(
        self,
        candidates: List[DecompositionCandidate],
        session: Session,
        iteration_id: int,
    ) -> List[GroupCategoryEntry]:
        """Create new GroupCategoryEntry records for decomposed components"""
        logger.info(f"Creating decomposed entries for {len(candidates)} candidates")

        new_entries = []

        # Group candidates by original entry to avoid duplicates
        entry_to_candidates = defaultdict(list)
        for candidate in candidates:
            entry_to_candidates[candidate.original_entry.id].append(candidate)

        for entry_id, entry_candidates in entry_to_candidates.items():
            # Use the best candidate for this entry
            best_candidate = max(entry_candidates, key=lambda c: c.overall_confidence)
            original_entry = best_candidate.original_entry

            # Create entries for each component
            for component in best_candidate.components:
                # Format component with proper title case
                formatted_component = format_tag_title_case(component)

                new_entry = GroupCategoryEntry(
                    folder_id=original_entry.folder_id,
                    partial_category_id=original_entry.partial_category_id,
                    group_id=original_entry.group_id,
                    iteration_id=iteration_id,
                    cluster_id=original_entry.cluster_id,
                    processed_name=formatted_component,
                    pre_processed_name=original_entry.pre_processed_name,
                    derived_names=(original_entry.derived_names or [])
                    + [best_candidate.original_tag],
                    path=original_entry.path,
                    confidence=original_entry.confidence
                    * best_candidate.overall_confidence,
                    processed=False,  # Will be processed in subsequent stages
                )
                new_entries.append(new_entry)
                session.add(new_entry)

            logger.debug(
                f"Multi-modal decomposition: '{best_candidate.original_tag}' -> {best_candidate.components}"
            )
            logger.debug(
                f"  Confidence: {best_candidate.overall_confidence:.3f}, Evidence: {best_candidate.evidence_types}"
            )
            logger.debug(
                f"  Scores - freq: {best_candidate.frequency_score:.3f}, embed: {best_candidate.embedding_score:.3f}, "
                f"pattern: {best_candidate.pattern_score:.3f}, cooc: {best_candidate.cooccurrence_score:.3f}"
            )

        return new_entries


def decompose_compound_tags(session: Session) -> None:
    """
    Main function to decompose compound tags in the most recent GroupCategoryEntry iteration.

    This function:
    1. Retrieves the most recent iteration of GroupCategoryEntry records
    2. Analyzes tags for potential decomposition using frequency and semantic analysis
    3. Creates new entries for identified component tags
    4. Updates the database with decomposed entries
    """
    logger.info("Starting compound tag decomposition")

    # Get the next iteration ID
    from grouping.group import get_next_iteration_id

    iteration_id = get_next_iteration_id(session)

    # Get entries from the previous iteration
    stmt = select(GroupCategoryEntry).where(
        GroupCategoryEntry.iteration_id == iteration_id - 1
    )
    entries = session.scalars(stmt).all()

    if not entries:
        logger.warning("No entries found for tag decomposition")
        return

    logger.info(f"Analyzing {len(entries)} entries for tag decomposition")

    # Initialize decomposer
    decomposer = TagDecomposer()

    # Extract comprehensive statistics and compute vectors/embeddings
    decomposer.component_stats = decomposer.extract_component_statistics(entries)
    decomposer.tag_vectors, decomposer.embeddings = (
        decomposer.compute_tag_vectors_and_embeddings()
    )

    logger.info(
        f"Found {len(decomposer.component_stats.word_frequencies)} unique component words "
        f"in {decomposer.component_stats.total_compounds} compound tags"
    )

    # Generate decomposition candidates
    candidates = decomposer.generate_decomposition_candidates(entries)

    if not candidates:
        logger.info("No valid decomposition candidates found")
        # Copy forward all entries for the next iteration
        for entry in entries:
            new_entry = GroupCategoryEntry(
                folder_id=entry.folder_id,
                partial_category_id=entry.partial_category_id,
                group_id=entry.group_id,
                iteration_id=iteration_id,
                cluster_id=entry.cluster_id,
                processed_name=entry.processed_name,
                pre_processed_name=entry.pre_processed_name,
                derived_names=entry.derived_names,
                path=entry.path,
                confidence=entry.confidence,
                processed=entry.processed,
            )
            session.add(new_entry)
    else:
        # Create decomposed entries
        new_entries = decomposer.create_decomposed_entries(
            candidates, session, iteration_id
        )

        # Copy forward entries that weren't decomposed
        decomposed_entry_ids = {candidate.original_entry.id for candidate in candidates}
        for entry in entries:
            if entry.id not in decomposed_entry_ids:
                new_entry = GroupCategoryEntry(
                    folder_id=entry.folder_id,
                    partial_category_id=entry.partial_category_id,
                    group_id=entry.group_id,
                    iteration_id=iteration_id,
                    cluster_id=entry.cluster_id,
                    processed_name=entry.processed_name,
                    pre_processed_name=entry.pre_processed_name,
                    derived_names=entry.derived_names,
                    path=entry.path,
                    confidence=entry.confidence,
                    processed=entry.processed,
                )
                session.add(new_entry)

        logger.info(f"Created {len(new_entries)} new decomposed entries")

    # Commit changes
    session.commit()

    logger.info(
        f"Tag decomposition completed. Added entries for iteration {iteration_id}"
    )


if __name__ == "__main__":
    # Example usage for testing
    import sys
    from pathlib import Path
    from data_models.database import get_sessionmaker

    if len(sys.argv) != 2:
        logger.error("Usage: python tag_decomposition.py <db_path>")
        sys.exit(1)

    db_path = Path(sys.argv[1])
    sessionmaker = get_sessionmaker(db_path)

    with sessionmaker() as session:
        decompose_compound_tags(session)
