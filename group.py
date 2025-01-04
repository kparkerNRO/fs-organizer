from collections import defaultdict
from typing import Optional
from dataclasses import dataclass
from rapidfuzz import fuzz
import re
import difflib
import sqlite3

from database import setup_group
from utils.filename_utils import (
    get_max_common_string,
    get_max_common_words,
    strip_part_from_base,
)
from nltk.metrics import edit_distance

common_words = {"the", "a", "an", "of", "in", "on", "at"}


def create_grouping(target_name, overlap_name):
    """
    Seperates out the common "overlap name" from the target name to create a
    heierachicial category, ensuring only words overlap
    """
    target_tokens = target_name.split()
    target_lower_tokens = target_name.lower().split()

    overlap_tokens = (
        overlap_name if isinstance(overlap_name, list) else (overlap_name.split())
    )
    overlap_tokens_lower = [token.lower() for token in overlap_tokens]

    if target_lower_tokens[: len(overlap_tokens)] == overlap_tokens_lower:
        if target_lower_tokens == overlap_tokens_lower:
            return [overlap_name]
        else:
            return [overlap_name, " ".join(target_tokens[len(overlap_tokens) :])]


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

                grouping = create_grouping(split_name, working_token)
                if grouping and is_valid_category(grouping[-1]):
                    names_to_replacement[next_name] = grouping
                    # names_to_process.remove(next_name)

            elif next_name_lower.startswith(first_token):
                # base case, initialize a new token
                working_token = overlap_func(tokens, next_name)

                # record that we have a matching token
                token_to_filenames[working_token] = {base_name, next_name}

                # store the new path name mapping
                grouping = create_grouping(next_name, working_token)
                if grouping and is_valid_category(grouping[-1]):
                    names_to_replacement[next_name] = grouping
                    # names_to_process.remove(next_name)

                grouping = create_grouping(base_name, working_token)
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


@dataclass
class GroupEntry:
    original_name: str
    grouped_name: str = ""
    categories: list[str] = ""
    dirty: bool = False
    confidence: float = 0.0


class Grouper:
    def __init__(self, group: list[str]):
        self.name_mapping: dict[str:GroupEntry] = {
            name: GroupEntry(name, grouped_name=name) for name in group
        }

    def update_mapping(
        self,
        new_mapping: dict[str, list[str]],
        confidence_mapping: dict[str, float] = {},
        clean_only: bool = False,
    ) -> list[str]:
        if new_mapping:
            for name, new_names in new_mapping.items():
                if name in self.name_mapping:
                    self.name_mapping[name].dirty = True
                    self.name_mapping[name].grouped_name = new_names[0]
                    self.name_mapping[name].categories = new_names
        if confidence_mapping:
            for name, confidence in confidence_mapping.items():
                if name in self.name_mapping:
                    self.name_mapping[name].confidence = confidence

        processed_mapping = [
            value.grouped_name
            for name, value in self.name_mapping.items()
            if not clean_only or not value.dirty
        ]
        return processed_mapping

    def get_changed_group_entries(self) -> dict[str, GroupEntry]:
        return {key: value for key, value in self.name_mapping.items() if value.dirty}

    def process_group(self):
        """ """
        working_list = list(self.name_mapping.keys())

        # check normalizing the names
        normalized_group = normalized_grouping(working_list)
        confidence_mapping = {name: 1.0 for name in working_list}
        working_list = self.update_mapping(
            normalized_group, confidence_mapping=confidence_mapping
        )

        corrected_spelling = spelling_grouping(working_list)
        confidence_mapping = {name: 0.95 for name in working_list}
        working_list = self.update_mapping(
            corrected_spelling, confidence_mapping=confidence_mapping, clean_only=True
        )

        if len(working_list) == 0:
            return

        # Now check for sub-grouping
        # check if there is a common prefix
        common_token = common_token_grouping(working_list)

        if common_token is None or len(common_token) == 0:
            return

        confidence_mapping = {name: 0.85 for name in working_list}

        # check the common tokens for possible mispellings - these will be the same length according to the grouping, and closely related
        new_name_size_to_name = defaultdict(list)
        for name in common_token.keys():
            new_name_size_to_name[len(common_token[name])].append(name)

        for size, names in new_name_size_to_name.items():
            if len(names) > 1:
                for i in range(size):
                    words_to_compare = [common_token[name][i] for name in names]
                    for j in range(len(words_to_compare)):
                        for k in range(len(words_to_compare)):
                            if j == k:
                                continue
                            if words_to_compare[j] == words_to_compare[k]:
                                continue

                            if (
                                edit_distance(
                                    words_to_compare[j],
                                    words_to_compare[k],
                                    transpositions=True,
                                )
                                < 3
                            ):
                                confidence_mapping[names[j]] = 0.6
                                confidence_mapping[names[k]] = 0.6
                            elif words_to_compare[j] in words_to_compare[k]:
                                confidence_mapping[names[j]] = 0.7
                                confidence_mapping[names[k]] = 0.7

        working_list = self.update_mapping(
            common_token, confidence_mapping=confidence_mapping, clean_only=True
        )

        # at this point we've exhausted all the high-confidence options, start moving into more esoteric options

        # if there are any unmatched names at this point, include them with low_confidence
        self.update_mapping(
            {name: [name] for name in working_list},
            confidence_mapping={name: 0.1 for name in working_list},
            clean_only=True,
        )


def group_by_algorithm(grouping_func, distinct_names):
    groups = []  # List to store groups of related terms

    def add_to_group(name: str):
        """Add a name to an existing group or create a new group."""
        for group in groups:
            # similarity = grouping_func(name, group)
            if any(grouping_func(name, member) for member in group):
                group.append(name)
                return
        groups.append([name])  # Create a new group if no match found

    for name in distinct_names:
        add_to_group(name)

    return groups


def generate_group_categories(db_cursor, groups, grouping_algorithm) -> set:
    processed_names = set()
    for group in groups:
        new_grouping = grouping_algorithm(group)
        if new_grouping:
            for name, new_names in new_grouping.items():
                db_cursor.execute(
                    """
                    INSERT INTO cleaned_groups (folder_name, grouped_name)
                    VALUES (?, ?)
                    """,
                    (name, ",".join(new_names)),
                )
            processed_names.update(new_grouping.keys())

    return processed_names


def calculate_similarity_difflib(a: str, b: str, threshold=80) -> bool:
    """Calculate similarity using difflib."""
    return (difflib.SequenceMatcher(None, a, b).ratio() * 100) >= threshold


def group_categories(db_path: str, threshold: int = 80):
    
    setup_group(db_path)
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # get the groups
    distinct_names = [
        row[0]
        for row in cur.execute(
            """
        SELECT DISTINCT category FROM categories
        """
        ).fetchall()
    ]

    groups = group_by_algorithm(calculate_similarity_difflib, distinct_names)
    filtered_groups = [group for group in groups if len(group) > 1]

    group_mapping = dict()

    for index, group in enumerate(filtered_groups):
        # create the group record
        group_name = f"group_{index}"
        group_mapping[index] = group
        cur.execute(
            """
                INSERT INTO groups (id,group_name, cannonical_name)
                VALUES (?, ?, ?)
                """,
            (index, group_name, group[0]),
        )

        for group_entry in group:
            cur.execute(
                """
                INSERT INTO processed_names (group_id, folder_name)
                VALUES (?, ?)
                """,
                (index, group_entry),
            )
    conn.commit()
    conn.close()

    return group_mapping


def process_groups(db_path: str, group_mapping: dict[int, list[str]]):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE processed_names
        SET grouped_name = null, confidence = null
        """
    )

    for _, group in group_mapping.items():
        # process the group
        grouper = Grouper(group)
        grouper.process_group()
        new_grouping = grouper.get_changed_group_entries()
        if new_grouping:
            for name, group_entry in new_grouping.items():
                cur.execute(
                    """
                    UPDATE processed_names
                    SET grouped_name = ?, confidence = ?
                    WHERE folder_name = ?
                    """,
                    (",".join(group_entry.categories), group_entry.confidence, name),
                )
            conn.commit()

    conn.commit()
    conn.close()


def calculate_and_process_groups(db_path: str, threshold: int = 80):
    group_mapping = group_categories(db_path, threshold)
    process_groups(db_path, group_mapping)
    return group_mapping


def process_pre_calculated_groups(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    groups = cur.execute(
        """
        SELECT group_id, folder_name
        FROM processed_names
        """
    ).fetchall()
    group_mapping = defaultdict(list)
    for group_id, folder_name in groups:
        group_mapping[group_id].append(folder_name)

    process_groups(db_path, group_mapping)
