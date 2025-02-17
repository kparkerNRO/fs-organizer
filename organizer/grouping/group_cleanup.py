from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from wordfreq import zipf_frequency
from grouping.helpers import (
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
        if name in filename_to_group_map and len(new_names) > 1:
            filename_to_group_map[name].dirty = True
            filename_to_group_map[name].grouped_name = new_names[0]
            filename_to_group_map[name].categories = new_names
            filename_to_group_map[name].confidence = confidence_value

    return working_list


def merge_groups(group1: str, group2: str, group_to_entries):
    for entry in group_to_entries[group2]:
        entry.grouped_name = group1
        entry.categories = [group1] + entry.categories[1:]
        group_to_entries[group1].append(entry)


def unify_category_spelling(group_to_entries: dict[str, list[GroupEntry]]):
    # normalize category names to use the same spelling

    for group_name in group_to_entries.keys():
        for compare_group in group_to_entries.keys():
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

    main_group = max(group_to_counts, key=group_to_counts.get)
    filenames_in_main = [entry.original_name for entry in group_to_entries[main_group]]
    for group, entries in group_to_entries.items():
        if group == main_group:
            continue

        group_entry = entries[0]
        edit_dists = [
            edit_distance(comp_file, group_entry.original_name, transpositions=True)
            for comp_file in filenames_in_main
        ]
        min_edit_dist = min(edit_dists)
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


def refine_groups(filenames: list[str]) -> dict[str, list[GroupEntry]]:
    filename_to_group_map: dict[str, GroupEntry] = {
        name: GroupEntry(name, grouped_name=name, confidence=1, categories=[name])
        for name in filenames
    }

    # normalize the names - correct for casing, punctuation, and spelling
    update_group_mapping(1.0, filename_to_group_map, normalized_grouping)
    update_group_mapping(0.95, filename_to_group_map, spelling_grouping)

    # group names by common tokens
    update_group_mapping(0.85, filename_to_group_map, common_token_grouping)

    # at this point they've been grouped, so any further processing is group-aware

    refined_groups = defaultdict(list)
    for group_entry in filename_to_group_map.values():
        refined_groups[group_entry.grouped_name].append(group_entry)

    if len(refined_groups) == 1:
        return refined_groups

    unify_category_spelling(refined_groups)
    process_outliers(refined_groups)
    process_ungrouped(refined_groups, filenames)

    # TODO - handle the "remainders" - names that don't fit into any group
    """
    probably this looks like:
        checking if the grouping consists of one large group, and 1+ groups with only one member
        store the single group memeber in the large group with a confidence equal 
            to the inverse of the edit distance

        after this processing, any remaining singleton lists should be flagged as low confidence
            maybe regroup them into a single group with low confidence if they are all singles
            (long term, this should handle suffix matching, but that's out of scope for)
    """

    return refined_groups


class Grouper:
    def __init__(self, group: list[str]):
        self.name_mapping: dict[str, GroupEntry] = {
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

    def get_group_name(self) -> str:
        """
        get the most common group name in the mapping
        """
        group_name = defaultdict(int)
        for name in self.name_mapping.values():
            group_name[name.grouped_name] += 1

        max_name = max(group_name, key=group_name.get)
        if group_name[max_name] > 1:
            return max_name

    def process_group(self):
        """Process the group using various grouping strategies"""
        working_list = list(self.name_mapping.keys())

        # Check normalizing the names
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

        # Check for sub-grouping and common prefix
        common_token = common_token_grouping(working_list)
        if not common_token:
            return

        confidence_mapping = {name: 0.85 for name in working_list}

        # Check common tokens for possible misspellings
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

        # Include remaining unmatched names with low confidence
        self.update_mapping(
            {name: [name] for name in working_list},
            confidence_mapping={name: 0.2 for name in working_list},
            clean_only=True,
        )
