from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from typing import Optional

from rapidfuzz.distance import Levenshtein
from sqlalchemy import select
from sqlalchemy.orm import Session
from wordfreq import zipf_frequency

from stages.grouping.helpers import (
    common_token_grouping,
    get_next_iteration_id,
    has_number_difference,
    normalized_grouping,
    spelling_grouping,
)
from storage.work_models import GroupCategoryEntry, GroupIteration

logger = getLogger(__name__)


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


def _update_group_mapping(confidence_value, filename_to_group_map, mapping_function):
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


def _group_by_tokens(
    filename_to_group_map: dict[str, GroupEntry], confidence_value: float = 0.85
):
    """
    common token grouping - has a different confidence evaluation so
    it can't be used with the generic update_group_mapping function
    """
    working_list = list(filename_to_group_map.keys())
    groups = common_token_grouping(working_list, prefer_longer_names=True)

    if not groups:
        return

    for name, new_names in groups.items():
        if name in filename_to_group_map and len(new_names) > 1:
            filename_to_group_map[name].dirty = True
            filename_to_group_map[name].grouped_name = new_names[0]
            filename_to_group_map[name].categories = new_names
            filename_to_group_map[name].confidence = confidence_value


def _unify_category_spelling(group_to_entries: dict[str, list[GroupEntry]]):
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

            if Levenshtein.distance(
                group_name, compare_group
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


def _process_outliers(group_to_entries: dict[str, list[GroupEntry]]):
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
            Levenshtein.distance(comp_file, group_entry.original_name)
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


def _process_ungrouped(
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

    # for group, entries in group_to_entries.items():
    #     entry = entries[0]
    #     if filename_to_count[entry.original_name] == 1:
    #         entry.confidence = 0.1


def refine_group(filenames: list[str]) -> dict[str, list[GroupEntry]]:
    filename_to_group_map: dict[str, GroupEntry] = {
        name: GroupEntry(name, grouped_name=name, confidence=1, categories=[name])
        for name in filenames
    }

    # normalize the names - correct for casing, punctuation, and spelling
    _update_group_mapping(1.0, filename_to_group_map, normalized_grouping)
    _update_group_mapping(0.95, filename_to_group_map, spelling_grouping)

    # group names by common tokens
    _group_by_tokens(filename_to_group_map)

    # at this point they've been grouped, so any further processing is group-aware

    refined_groups = defaultdict(list)
    for group_entry in filename_to_group_map.values():
        refined_groups[group_entry.grouped_name].append(group_entry)

    if len(refined_groups) == 1:
        return refined_groups

    _unify_category_spelling(refined_groups)
    _process_outliers(refined_groups)
    _process_ungrouped(refined_groups, filenames)

    return refined_groups


def apply_group_cleanup(
    session: Session,
    run_id: int,
    snapshot_id: int,
) -> None:
    """
    Apply group cleanup to refine and merge similar group names.
    This step processes groups from the previous iteration and applies normalization,
    spelling correction, and similarity-based merging to clean up group names.
    """
    logger.info("Applying group cleanup")

    # Get the current iteration ID and create a new iteration
    iteration_id = get_next_iteration_id(session)

    iteration = GroupIteration(
        id=iteration_id,
        run_id=run_id,
        snapshot_id=snapshot_id,
        description="Group cleanup and refinement",
    )
    session.add(iteration)
    session.flush()

    # Get entries from the previous iteration
    stmt = select(GroupCategoryEntry).where(
        GroupCategoryEntry.iteration_id == iteration_id - 1
    )
    previous_entries = session.scalars(stmt).all()

    if not previous_entries:
        logger.info("No entries to process")
        return

    # Group entries by folder_id
    folder_to_entries = defaultdict(list)
    for entry in previous_entries:
        folder_to_entries[entry.folder_id].append(entry)

    # Process each folder's entries
    for folder_id, entries in folder_to_entries.items():
        # Extract the processed names for this folder
        names = [entry.processed_name or entry.pre_processed_name for entry in entries]
        names = [name for name in names if name]  # Filter out None/empty

        if not names:
            continue

        # Apply the refinement logic
        refined_groups = refine_group(names)

        # Create a mapping from original name to entry for quick lookup
        name_to_entry = {
            entry.processed_name or entry.pre_processed_name: entry for entry in entries
        }

        # Create new entries based on refined groups
        for group_name, group_entries in refined_groups.items():
            for group_entry in group_entries:
                original_entry = name_to_entry.get(group_entry.original_name)
                if not original_entry:
                    continue

                # Create entries for all categories if they exist
                if group_entry.categories and len(group_entry.categories) > 1:
                    # Multiple categories - create an entry for each
                    for category in group_entry.categories:
                        new_entry = GroupCategoryEntry(
                            folder_id=folder_id,
                            iteration_id=iteration_id,
                            pre_processed_name=original_entry.pre_processed_name,
                            processed_name=category,
                            path=original_entry.path,
                            confidence=group_entry.confidence,
                            processed=False,
                            derived_names=original_entry.derived_names,
                        )
                        session.add(new_entry)
                else:
                    # Single category or grouped name
                    new_entry = GroupCategoryEntry(
                        folder_id=folder_id,
                        iteration_id=iteration_id,
                        pre_processed_name=original_entry.pre_processed_name,
                        processed_name=group_entry.grouped_name,
                        path=original_entry.path,
                        confidence=group_entry.confidence,
                        processed=False,
                        derived_names=original_entry.derived_names,
                    )
                    session.add(new_entry)

    session.commit()
    logger.info("Group cleanup complete")
