import difflib
from enum import Enum
import re
from typing import Optional

from nltk.metrics import edit_distance
from rapidfuzz import fuzz

from utils.filename_utils import (
    clean_filename,
    get_max_common_words,
    strip_part_from_base,
)

common_words = {"the", "a", "an", "of", "in", "on", "at"}


def split_category(target_name, category, clean_result: bool = True):
    """
    Seperates out the common "overlap name" from the target name to create a
    heierachicial category, ensuring only words overlap
    """
    target_tokens = target_name.split()
    target_lower_tokens = target_name.lower().split()

    overlap_tokens = category if isinstance(category, list) else (category.split())
    overlap_tokens_lower = [token.lower() for token in overlap_tokens]

    if clean_result:
        category = new_name = clean_filename(category)

    if target_lower_tokens[: len(overlap_tokens)] == overlap_tokens_lower:
        if target_lower_tokens == overlap_tokens_lower:
            return [category]
        else:
            new_name = " ".join(target_tokens[len(overlap_tokens) :])
            if clean_result:
                new_name = clean_filename(new_name)
            return [category, new_name]


def normalize_for_comparison(s: str) -> str:
    """Normalize string for comparison while preserving meaningful differences."""
    # Convert to lowercase
    s = s.lower()
    # Remove articles and common words from start
    s = re.sub(r"^(the|a|an)\s+", "", s)
    # Remove apostrophes and normalize spaces
    s = re.sub(r"[\'\']s?\b", "", s)
    return " ".join(s.split())


def normalized_grouping(group: list) -> Optional[dict[str, list[str]]]:
    new_grouping = dict()
    for category in group:
        if category in new_grouping:
            continue

        for category2 in group:
            if category != category2:
                if normalize_for_comparison(category) == normalize_for_comparison(
                    category2
                ):
                    if category[0].isupper():
                        new_grouping[category] = [category]
                        new_grouping[category2] = [category]
                    else:
                        new_grouping[category] = [category2]
                        new_grouping[category2] = [category2]

    keys = list(new_grouping.keys())
    new_grouping = {key: new_grouping[key] for key in keys if new_grouping[key]}
    if len(new_grouping) == 1:
        return None

    return new_grouping


def has_number_difference(str1: str, str2: str) -> bool:
    """
    Check if the main difference between strings is numerical.
    Returns True if strings are similar except for numbers.
    """
    nums1 = set(re.findall(r"\d+", str1))
    nums2 = set(re.findall(r"\d+", str2))

    # If both strings have numbers but they're different
    if nums1 and nums2 and nums1 != nums2:
        # Remove numbers and compare the rest
        base1 = re.sub(r"\d+", "", str1)
        base2 = re.sub(r"\d+", "", str2)

        # If the non-numeric parts are very similar
        if fuzz.ratio(base1, base2) > 95:
            return True

    return False


def spelling_grouping(group: list) -> Optional[dict[str, list[str]]]:
    new_grouping = dict()
    for category in group:
        if category in new_grouping:
            continue

        for category2 in group:
            if category != category2:
                if edit_distance(
                    category, category2, transpositions=True
                ) < 2 and not has_number_difference(category, category2):
                    new_grouping[category2] = [category]
                    new_grouping[category] = [category]

    keys = list(new_grouping.keys())
    new_grouping = {key: new_grouping[key] for key in keys if new_grouping[key]}
    if len(new_grouping) == 1:
        return None

    return new_grouping


def common_token_grouping(
    names_to_group: list, overlap_func=get_max_common_words
) -> Optional[dict[str, list[str]]]:
    if len(names_to_group) == 0:
        return None

    names_to_process = list(names_to_group)
    token_to_filenames = dict()
    names_to_replacement = dict()

    def is_valid_category(name: Optional[str]) -> bool:
        """Check if a category is invalid for grouping."""
        if name is None:
            return False
        too_short = len(name) < 2
        number_only = all(char.isdigit() for char in name)
        return not (too_short or number_only)

    for base_name in names_to_group:
        if base_name not in names_to_process:
            continue

        names_to_process.remove(base_name)

        tokens = base_name.split()
        first_token = tokens[0].lower()
        working_token = None

        for next_name in names_to_process:
            # if next_name in names_to_replacement:
            #     continue
            next_name_lower = next_name.lower()

            if working_token and next_name_lower.startswith(working_token):
                # main case: store a new token match
                token_to_filenames[working_token].add(next_name)
                split_name = strip_part_from_base(next_name, working_token)

                grouping = split_category(split_name, working_token)
                if grouping and is_valid_category(grouping[-1]):
                    names_to_replacement[next_name] = grouping
                    # names_to_process.remove(next_name)

            elif next_name_lower.startswith(first_token):
                # base case, initialize a new token
                working_token = overlap_func(tokens, next_name)

                # record that we have a matching token
                token_to_filenames[working_token] = {base_name, next_name}

                # store the new path name mapping
                grouping = split_category(next_name, working_token)
                if grouping and is_valid_category(grouping[-1]):
                    names_to_replacement[next_name] = grouping
                    # names_to_process.remove(next_name)

                grouping = split_category(base_name, working_token)
                if grouping and is_valid_category(grouping[-1]):
                    names_to_replacement[base_name] = grouping

    # at the end of the set, see if we accidentally grabbed two overlapping sets
    tokens_to_revisit = {}
    for token1 in token_to_filenames.keys():
        for token2 in token_to_filenames.keys():
            # if this isn't itself, and the main token is a subset of the other token
            if token1 in token2 and not token1 == token2:
                # replace the longer token with the shorter one in the replacement array
                delta = token2.replace(token1, "")
                for filename in token_to_filenames[token2]:
                    working_path = names_to_replacement[filename]
                    if working_path[0] == token1:
                        continue
                    if len(working_path) == 1:
                        continue
                    working_path[-1] = (delta + " " + working_path[-1]).strip()
                    working_path[-2] = token1

                # empty the longer token and transfer it's contents to the shorter one
                token_to_filenames[token1] |= token_to_filenames[token2]
                token_to_filenames[token2] = set()
                tokens_to_revisit[token1] = token_to_filenames[token1]

    # remove any empty entries
    names = list(names_to_replacement.keys())
    for name in names:
        if names_to_replacement[name] is None:
            del names_to_replacement[name]

    # if it's only one, it's not a grouping
    if len(names_to_replacement) == 1:
        return None

    return names_to_replacement


def calculate_similarity_difflib(a: str, b: str, threshold=80) -> bool:
    """Calculate similarity using difflib."""
    return (difflib.SequenceMatcher(None, a, b).ratio() * 100) >= threshold


class ClassificationType(str, Enum):
    VARIANT = "variant"
    CATEGORY = "category"
    SUBJECT = "subject"
    UNKNOWN = "unknown"
    CLUSTERED = "clustered"
