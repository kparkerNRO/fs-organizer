from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import (
    setup_folder_categories,
    setup_group,
    get_session,
    setup_category_summarization,
    Folder,
    FolderCategory,
    GroupRecord,
    ProcessedName,
    Category,
)
from grouping.folder_group import process_folders
from grouping.helpers import (
    ClassificationType,
    calculate_similarity_difflib,
    common_token_grouping,
    normalized_grouping,
    spelling_grouping,
)
from grouping.nlp_grouping import group_uncertain
from utils.config import KNOWN_VARIANT_TOKENS
from utils.filename_utils import (
    clean_filename,
    split_view_type,
)
from nltk.metrics import edit_distance
from pathlib import Path


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
            confidence_mapping={name: 0.1 for name in working_list},
            clean_only=True,
        )


def group_categories(db_path: Path, threshold: int = 80) -> dict[int, list[str]]:
    """Group categories using SQLAlchemy"""
    setup_group(db_path)
    session = get_session(db_path)

    try:
        # Get distinct categories
        distinct_categories = (
            session.query(FolderCategory.original_name).distinct().all()
        )
        distinct_names = [cat[0] for cat in distinct_categories]

        groups = []  # List to store groups of related terms
        for name in distinct_names:
            # Try to add to existing group or create new one
            for group in groups:
                if any(
                    calculate_similarity_difflib(name, member, threshold)
                    for member in group
                ):
                    group.append(name)
                    break
            else:
                groups.append([name])

        filtered_groups = [group for group in groups if len(group) > 1]
        group_mapping = {}

        for index, group in enumerate(filtered_groups):
            group_name = f"group_{index}"
            group_mapping[index] = group

            # Create group record
            new_group = GroupRecord(
                id=index, group_name=group_name, cannonical_name=group[0]
            )
            session.add(new_group)

            # Create processed names records
            for group_entry in group:
                processed_name = ProcessedName(group_id=index, folder_name=group_entry)
                session.add(processed_name)

        session.commit()
        return group_mapping
    finally:
        session.close()


def process_groups(db_path: Path, group_mapping: dict[int, list[str]]) -> None:
    """Process groups using SQLAlchemy"""
    session = get_session(db_path)

    try:
        # Reset processed names
        session.query(ProcessedName).update(
            {ProcessedName.grouped_name: None, ProcessedName.confidence: None}
        )

        # Process each group
        for _, group in group_mapping.items():
            grouper = Grouper(group)
            grouper.process_group()
            new_grouping = grouper.get_changed_group_entries()

            if new_grouping:
                for name, group_entry in new_grouping.items():
                    session.query(ProcessedName).filter(
                        ProcessedName.folder_name == name
                    ).update(
                        {
                            ProcessedName.grouped_name: ",".join(
                                group_entry.categories
                            ),
                            ProcessedName.confidence: group_entry.confidence,
                        }
                    )
                session.commit()

    finally:
        session.close()


def calculate_and_process_groups(
    db_path: Path, threshold: int = 80
) -> dict[int, list[str]]:
    """Calculate and process groups using SQLAlchemy"""
    group_mapping = group_categories(db_path, threshold)
    process_groups(db_path, group_mapping)
    return group_mapping


def process_pre_calculated_groups(db_path: Path) -> None:
    """Process pre-calculated groups using SQLAlchemy"""
    session = get_session(db_path)

    try:
        # Get existing groups
        groups = session.query(ProcessedName.group_id, ProcessedName.folder_name).all()

        group_mapping = defaultdict(list)
        for group_id, folder_name in groups:
            group_mapping[group_id].append(folder_name)

        process_groups(db_path, group_mapping)
    finally:
        session.close()
