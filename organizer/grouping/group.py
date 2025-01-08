from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from pipeline.database import (
    setup_folder_categories,
    setup_group,
    get_session,
    setup_category_summarization,
    Folder,
    FolderCategory,
    GroupRecord,
    Category,
)
from grouping.folder_group import process_folders
from grouping.group_cleanup import Grouper
from grouping.helpers import (
    ClassificationType,
    calculate_similarity_difflib,
    common_token_grouping,
    normalized_grouping,
    spelling_grouping,
)
from pipeline.nlp_grouping import cluster_with_custom_metric, group_uncertain
from utils.config import KNOWN_VARIANT_TOKENS
from utils.filename_utils import (
    clean_filename,
    split_view_type,
)
from nltk.metrics import edit_distance
from pathlib import Path


def parse_name(name: str) -> tuple[list[str], list[str]]:
    """Parse a name into categories and variants"""
    categories = []
    variants = []

    clean_row = clean_filename(name)
    components = clean_row.split("-")
    cleaned_components = [clean_filename(component) for component in components]

    for component in cleaned_components:
        category_current = component
        while category_current:
            category, variant = split_view_type(category_current, KNOWN_VARIANT_TOKENS)
            if category and category not in categories:
                if category in KNOWN_VARIANT_TOKENS:
                    variants.append(category)
                else:
                    categories.append(category)
            if variant:
                variants.append(variant)

            if category_current == category:
                break

            category_current = category

    return categories, variants


def heuristic_categorize(db_path: Path, update_table: bool = False) -> None:
    """
    Break the filenames into tokens, and identify known variants and suspected categories.
    Clean up the tokens, and assign the folder name to the first category or variant
    """

    session = get_session(db_path)

    try:
        # Get all folders
        folders = session.query(Folder).all()

        for folder in folders:
            name = folder.folder_name

            if name.endswith(".zip"):
                categories, variants = parse_name(name[:-4])
            else:
                categories, variants = parse_name(name)

            variants = list(set(variants))
            cleaned_name = (
                categories[0] if categories else variants[0] if variants else ""
            )

            # Update the folder
            folder.cleaned_name = cleaned_name
            folder.variants = variants
            folder.categories = categories

            if len(categories) == 0 and len(variants) > 0:
                folder.classification = ClassificationType.VARIANT
            elif len(categories) == 1 and len(variants) == 0:
                folder.classification = ClassificationType.CATEGORY
            else:
                folder.classification = ClassificationType.UNKNOWN

            for category in categories:
                category_lookup = FolderCategory(
                    folder_id=folder.id,
                    original_name=folder.folder_name,
                    name=category,
                    classification=folder.classification,
                )
                session.add(category_lookup)

            for variant in variants:
                category_lookup = FolderCategory(
                    folder_id=folder.id,
                    original_name=folder.folder_name,
                    name=variant,
                    classification=ClassificationType.VARIANT,
                )
                session.add(category_lookup)
        session.commit()

        non_variant_folders = (
            session.query(Folder)
            .filter(Folder.classification != ClassificationType.VARIANT)
            .all()
        )
        # for each of these folders, check if
        # (a) there is only one category and (b) that all sub-folders are classified as variant
        for folder in non_variant_folders:
            if len(folder.categories) == 1:
                children = (
                    session.query(Folder)
                    .filter(Folder.parent_path.startswith(folder.folder_path))
                    .all()
                )
                if len(children) == 0 or all(
                    child.classification == ClassificationType.VARIANT
                    for child in children
                ):
                    folder.classification = ClassificationType.SUBJECT
                    folder.subject = folder.categories[0]
                    # also update the FolderCategory reference for folder.categories[0]
                    category_lookup = (
                        session.query(FolderCategory)
                        .filter_by(folder_id=folder.id, name=folder.categories[0])
                        .one()
                    )
                    category_lookup.classification = ClassificationType.SUBJECT
        session.commit()

    finally:
        session.close()


def evaluate_categorization(db_path: Path) -> None:
    session = get_session(db_path)

    try:
        """
        Populate the WorkingCategory table from the Folder table
            category_name: the unique name of the category
            classification_counts: a map of classification types to counts
        """
        # Get all folders
        folder_category = (
            session.query(FolderCategory)
            # .filter(Folder.classification != ClassificationType.VARIANT)
            .all()
        )

        # Create a dictionary to store category counts
        category_counts = defaultdict(lambda: defaultdict(int))

        for category in folder_category:
            category_counts[category.name][category.classification] += 1

        # Insert into WorkingCategory table
        for category, counts in category_counts.items():
            categorization = ClassificationType.UNKNOWN
            if len(counts) == 1:
                categorization = list(counts.keys())[0]

            total_count = sum(counts.values())

            working_category = Category(
                category_name=category,
                classification_counts=dict(counts),
                classification=categorization,
                total_count=total_count,
            )
            session.add(working_category)

        session.commit()

    finally:
        session.close()


def consolidate_groups(db_path: Path) -> None:
    session = get_session(db_path)

    try:
        # Get all groups
        group_categories = (
            session.query(GroupRecord, FolderCategory)
            .join(FolderCategory, GroupRecord.category_id == FolderCategory.id)
            .all()
        )

        # Create a dictionary to store group counts
        group_unique_names = defaultdict(set)
        group_record_map = defaultdict(list)
        group_category_map = defaultdict(list)

        for group, category in group_categories:
            group_unique_names[group.group_id].add(group.cannonical_name)
            group_record_map[group.group_id].append(group)
            group_category_map[group.group_id].append(category)

        # Find groups with only one unique name
        singleton_groups = {
            group_id: list(group)[0]
            for group_id, group in group_unique_names.items()
            if len(group) == 1
        }

        for group_id, group_cannon_name in singleton_groups.items():
            # update the group records with the new name
            for group in group_record_map[group_id]:
                group.group_name = group_cannon_name
                group.processed = True

        #     if len(group_record_map[group_id]) == 1:
        #         continue
        #     group = group_record_map[group_id][0]
        #     existing_classifications = {
        #         category.classification for category in group_category_map[group_id]
        #     }
        #     classification = (
        #         list(existing_classifications)[0]
        #         if len(existing_classifications) == 1
        #         else ClassificationType.UNKNOWN
        #     )

        #     # create a new folder category record
        #     category_lookup = FolderCategory(
        #         folder_id=group.folder_id,
        #         name=group.cannonical_name,
        #         classification=classification,
        #         group_name=group_id,
        #     )
        #     session.add(category_lookup)

        #     # update the existing classifications to be hidden
        #     for category in group_category_map[group_id]:
        #         category.hidden = True
        #         category.group_name = group_id

        # for the rest of the groups, try to group them by common name
        remaining_groups = {
            group_id: group
            for group_id, group in group_record_map.items()
            if group_id not in singleton_groups
        }
        for group_id, group in remaining_groups.items():
            record_names = [record.cannonical_name for record in group]
            cannon_group_map = defaultdict(list)
            for record in group:
                cannon_group_map[record.cannonical_name].append(record)
            grouper = Grouper(record_names)
            grouper.process_group()
            new_grouping = grouper.get_changed_group_entries()
            for name, group_entry in new_grouping.items():
                if name in cannon_group_map:
                    for record in cannon_group_map[name]:
                        record.processed_names = group_entry.categories
                        # record.group_name = group_entry.grouped_name
                        record.processed = True
                        record.confidence = group_entry.confidence

        session.commit()

    finally:
        session.close()


def categorize(db_path: Path):
    setup_folder_categories(db_path)
    setup_category_summarization(db_path)
    setup_group(db_path)

    heuristic_categorize(db_path)
    evaluate_categorization(db_path)

    cluster_with_custom_metric(db_path)
    consolidate_groups(db_path)

    # first, process folders in the same folder group and pull out duplicate terms
    # process_folders(session)
