from collections import defaultdict
from typing import Optional
from dataclasses import dataclass
import sqlite3

from database import setup_group
from grouping.helpers import (
    calculate_similarity_difflib,
    common_token_grouping,
    normalized_grouping,
    spelling_grouping,
)
from utils.config import KNOWN_VARIANT_TOKENS
from utils.filename_utils import (
    clean_filename,
    split_view_type,
)
from nltk.metrics import edit_distance


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


def clean_file_name_post(db_path, update_table: bool = False):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    rows = [
        (row[0], row[1])
        for row in cur.execute(
            """
        SELECT id, folder_name FROM folders
        """
        ).fetchall()
    ]

    def parse_name(name):
        # do an initial clean up
        categories = []
        variants = []

        clean_row = clean_filename(name)

        # handle any sub-categories in the name
        components = clean_row.split("-")
        cleaned_components = [clean_filename(component) for component in components]

        for component in cleaned_components:
            category_current = component
            while category_current:
                category, variant = split_view_type(
                    category_current, KNOWN_VARIANT_TOKENS
                )
                if category and category not in categories:
                    categories.append(category)
                if variant:
                    variants.append(variant)

                if category_current == category:
                    break

                category_current = category

        return categories, variants

    for id, row in rows:
        if row[-4:] == ".zip":
            categories, variants = parse_name(row[:-4])

        else:
            categories, variants = parse_name(row)

        variants = list(set(variants))
        cleaned_name = categories[0] if categories else variants[0] if variants else ""

        cur.execute(
            """
            UPDATE folders
            SET cleaned_name = ?
            WHERE id = ?
        """,
            (cleaned_name, id),
        )

        # if categories:
        #     for category in categories:
        #         cur.execute(
        #             """
        #             INSERT INTO categories (category, folder_id)
        #             VALUES (?, ?)
        #             """,
        #             (category, id),
        #         )

    conn.commit()
    conn.close()


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

    groups = []  # List to store groups of related terms

    def add_to_group(name: str):
        """Add a name to an existing group or create a new group."""
        for group in groups:
            # similarity = grouping_func(name, group)
            if any(calculate_similarity_difflib(name, member) for member in group):
                group.append(name)
                return
        groups.append([name])  # Create a new group if no match found

    for name in distinct_names:
        add_to_group(name)

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

    # reset the table before going to work
    cur.execute(
        """
        UPDATE processed_names
        SET grouped_name = null, confidence = null
        """
    )

    # first, process folders in the same folder group and pull out duplicate terms

    # second, group by category

    # finally, correction and any other grouping options
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
