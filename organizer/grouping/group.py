from collections import defaultdict

from data_models.classify import ClassificationType
from data_models.database import (
    GroupCategory,
    setup_folder_categories,
    setup_group,
    get_session,
    setup_category_summarization,
    Folder,
    PartialNameCategory,
    GroupCategoryEntry,
    Category,
)
from grouping.group_cleanup import GroupEntry, Grouper

from pipeline.nlp_grouping import cluster_with_custom_metric
from utils.config import KNOWN_VARIANT_TOKENS
from utils.filename_utils import (
    clean_filename,
    split_view_type,
)
from pathlib import Path


def parse_name(name: str) -> tuple[list[str], list[str]]:
    """
    Parse out known names and suspected components
    """
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

    Fills out the "partial name category"
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
                category_lookup = PartialNameCategory(
                    folder_id=folder.id,
                    original_name=folder.folder_name,
                    name=category,
                    classification=folder.classification,
                )
                session.add(category_lookup)

            for variant in variants:
                category_lookup = PartialNameCategory(
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
                        session.query(PartialNameCategory)
                        .filter_by(folder_id=folder.id, name=folder.categories[0])
                        .one()
                    )
                    category_lookup.classification = ClassificationType.SUBJECT
        session.commit()

    finally:
        session.close()


def evaluate_categorization(db_path: Path) -> None:
    """
    Evaluate the partial name categories, and agregate the counts for each category
    """
    session = get_session(db_path)

    try:
        """
        Populate the WorkingCategory table from the Folder table
            category_name: the unique name of the category
            classification_counts: a map of classification types to counts
        """
        # Get all folders
        folder_category = (
            session.query(PartialNameCategory)
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
    """
    After groupings have been calculated, evalute the groups to determine which represent
    "real" groupings (i.e. single unique name, or multiple names with a common prefix), and
    which are unable to be grouped.
    
    """
    session = get_session(db_path)

    try:
        # Get all processed categories
        group_categories: list[GroupCategoryEntry] = (
            session.query(GroupCategoryEntry)
            .all()
        )

        group_record_map = defaultdict(list)

        calculated_groups = defaultdict(list)

        for group in group_categories:
            # group_unique_names[group.group_id].add(group.original_name)
            group_record_map[group.group_id].append(group)
            # group_category_map[group.group_id].append(category)

        for group_id, group_items in group_record_map.items():
            
            # singletons - just store them and move on
            if len(group_items) == 1:
                calculated_groups[group_items[0].original_name].append(group_items[0])
                group_items[0].processed = True
                group_items[0].confidence = 1.0
                continue

            record_names = [record.original_name for record in group_items]

            # evaluate the group - this does a few things
            # 1. try to normalize the names to a common prefix correcting for spelling
            # 2. determine sub-categories when the group name is a common prefix
            grouper = Grouper(record_names)
            grouper.process_group()

            name_to_processed_entry: dict[str,list[GroupEntry]] = defaultdict(list)
            for record in group_items:
                record.processed = True
                name_to_processed_entry[record.original_name].append(record)

            new_grouping = grouper.name_mapping
            grouped_name = grouper.get_group_name()
            if grouped_name is not None:
                for name, group_entry in new_grouping.items():
                    if name in name_to_processed_entry:
                        for record in name_to_processed_entry[name]:
                            record.derived_names = group_entry.categories
                            # record.group_name = group_entry.grouped_name
                            record.processed = True
                            record.confidence = group_entry.confidence

                            calculated_groups[grouped_name].append(record)
      
        i=0
        for group_id, group_items in calculated_groups.items():
            group_confidence = min([record.confidence for record in group_items])
            db_category = GroupCategory(id=i, name=group_id, count=len(group_items), group_confidence=group_confidence)
            session.add(db_category)
            for group_item in group_items:
                group_item.group_id = i
            i += 1

        session.commit()

    finally:
        session.close()


def categorize(db_path: Path):
    
    # setup the database
    setup_folder_categories(db_path)
    setup_category_summarization(db_path)
    setup_group(db_path)


    heuristic_categorize(db_path)
    evaluate_categorization(db_path)

    cluster_with_custom_metric(db_path)
    consolidate_groups(db_path)
