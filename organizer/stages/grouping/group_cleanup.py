from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from wordfreq import zipf_frequency
from stages.grouping.helpers import (
    common_token_grouping,
    has_number_difference,
    normalized_grouping,
    spelling_grouping,
)

from nltk.metrics import edit_distance


@dataclass
class GroupEntry:
    original_name: str
    grouped_name: str = ""
    categories: Optional[list[str]] = None
    dirty: bool = False
    confidence: float = 0.0

    def __post_init__(self):
        if self.categories is None:
            self.categories = []


def update_group_mapping(confidence_value, filename_to_group_map, mapping_function):
    working_list = list(filename_to_group_map.keys())
    group = mapping_function(working_list)

    if not group:
        return

    for name, new_names in group.items():
        filename_to_group_map[name].dirty = True
        filename_to_group_map[name].grouped_name = new_names[0]
        filename_to_group_map[name].categories = new_names
        filename_to_group_map[name].confidence = confidence_value

    return working_list


def group_by_tokens(
    filename_to_group_map: dict[str, GroupEntry], confidence_value: float = 0.85
):
    """
    common token grouping - has a different confidence evaluation so
    it can't be used with the generic update_group_mapping function
    """
    working_list = list(filename_to_group_map.keys())
    groups = common_token_grouping(working_list)

    if not groups:
        return

    for name, new_names in groups.items():
        if name in filename_to_group_map and len(new_names) > 1:
            filename_to_group_map[name].dirty = True
            filename_to_group_map[name].grouped_name = new_names[0]
            filename_to_group_map[name].categories = new_names
            filename_to_group_map[name].confidence = confidence_value


def unify_category_spelling(group_to_entries: dict[str, list[GroupEntry]]):
    # normalize category names to use the same spelling
    def merge_groups(group1: str, group2: str, group_to_entries):
        group2_entries = group_to_entries.get(group2, [])
        group1_entries = group_to_entries.get(group1, [])

        for entry in group2_entries:
            entry.grouped_name = group1
            if entry.categories:
                entry.categories = [group1] + entry.categories[1:]
            else:
                entry.categories = [group1]
            group1_entries.append(entry)

    group_keys = list(group_to_entries.keys())
    for group_name in group_keys:
        for compare_group in group_keys:
            if group_name == compare_group:
                continue

            if edit_distance(
                group_name, compare_group, transpositions=True
            ) < 3 and not has_number_difference(group_name, compare_group):
                # see if one of them is a real word
                freq_group = zipf_frequency(group_name, "en")
                freq_compare_group = zipf_frequency(compare_group, "en")
                new_group = None
                if freq_group > freq_compare_group:
                    merge_groups(group_name, compare_group, group_to_entries)
                    new_group = group_name
                elif freq_compare_group > freq_group:
                    merge_groups(compare_group, group_name, group_to_entries)
                    new_group = group_name
                elif freq_group == freq_compare_group:
                    # if they are the same frequency, just pick the first one
                    merge_groups(group_name, compare_group, group_to_entries)
                    new_group = group_name

                if new_group:
                    for entry in group_to_entries[new_group]:
                        entry.confidence = 0.6


def process_outliers(group_to_entries: dict[str, list[GroupEntry]]):
    """
    Process the outliers - groups with only one member
    """
    if len(group_to_entries) == 1:
        return

    group_to_counts = {
        group: len(entries) for group, entries in group_to_entries.items()
    }

    """
    The most expected case here is that all but one was grouped into
    a single group. In this case, we should add the remaining group to the
    main group with a confidence equal to the inverse of the edit distance
    """

    is_single_group = (
        len([count for count in group_to_counts.values() if count > 1]) == 1
    )
    is_only_remainders = (
        len([count for count in group_to_counts.values() if count == 1])
        == len(group_to_counts) - 1
    )

    if not (is_single_group and is_only_remainders):
        return

    main_group = max(group_to_counts, key=group_to_counts.__getitem__)
    filenames_in_main = [entry.original_name for entry in group_to_entries[main_group]]
    for group, entries in group_to_entries.items():
        if group == main_group:
            continue

        group_entry = entries[0]
        edit_dists = [
            edit_distance(comp_file, group_entry.original_name, transpositions=True)
            for comp_file in filenames_in_main
        ]
        min_edit_dist = min(edit_dists) or 1
        confidence = 1 / min_edit_dist
        group_entry.grouped_name = main_group
        group_entry.confidence = confidence
        group_to_entries[main_group].append(group_entry)
        group_to_entries[group].pop()

    groups = list(group_to_entries.keys())
    for group in groups:
        if len(group_to_entries[group]) == 0:
            del group_to_entries[group]


def process_ungrouped(
    group_to_entries: dict[str, list[GroupEntry]], filenames: list[str]
):
    """
    Process the remaining ungrouped entries
    """
    if len(group_to_entries) == 1:
        return group_to_entries

    only_singles = all(len(entries) == 1 for entries in group_to_entries.values())

    if not only_singles:
        return

    filename_to_count = defaultdict(int)
    for filename in filenames:
        filename_to_count[filename] += 1

    for group, entries in group_to_entries.items():
        entry = entries[0]
        if filename_to_count[entry.original_name] == 1:
            entry.confidence = 0.1


def refine_group(filenames: list[str]) -> dict[str, list[GroupEntry]]:
    filename_to_group_map: dict[str, GroupEntry] = {
        name: GroupEntry(name, grouped_name=name, confidence=1, categories=[name])
        for name in filenames
    }

    # normalize the names - correct for casing, punctuation, and spelling
    update_group_mapping(1.0, filename_to_group_map, normalized_grouping)
    update_group_mapping(0.95, filename_to_group_map, spelling_grouping)

    # group names by common tokens
    group_by_tokens(filename_to_group_map)

    # at this point they've been grouped, so any further processing is group-aware

    refined_groups = defaultdict(list)
    for group_entry in filename_to_group_map.values():
        refined_groups[group_entry.grouped_name].append(group_entry)

    if len(refined_groups) == 1:
        return refined_groups

    unify_category_spelling(refined_groups)
    process_outliers(refined_groups)
    process_ungrouped(refined_groups, filenames)

    return refined_groups
