from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy.orm import Session

from data_models.classify import ClassificationType
from data_models.database import (
    GroupCategory,
    # setup_category_summarization,
    setup_folder_categories,
    setup_group,
    get_session,
    get_sessionmaker,
    Folder,
    PartialNameCategory,
    GroupCategoryEntry,
)
from grouping.group_cleanup import refine_group
from logging import getLogger
from grouping.nlp_grouping import (
    ClusterItem,
    cluster_with_custom_metric,
    prepare_records,
    TEXT_DISTANCE_RATIO,
)
from utils.config import KNOWN_VARIANT_TOKENS
from utils.filename_utils import (
    clean_filename,
    split_view_type,
)
from pathlib import Path

REVIEW_CONFIDENCE_THRESHOLD = 0.7
log = getLogger(__name__)

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


def heuristic_categorize(session) -> None:
    """
    Break the filenames into tokens, and identify known variants and suspected categories.
    Clean up the tokens, and assign the folder name to the first category or variant

    Fills out the "partial name category"
    """

    # Get all folders
    folders = session.query(Folder).all()

    for folder in folders:
        name = folder.folder_name

        if name.endswith(".zip"):
            categories, variants = parse_name(name[:-4])
        else:
            categories, variants = parse_name(name)

        variants = list(set(variants))
        cleaned_name = categories[0] if categories else variants[0] if variants else ""

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
                child.classification == ClassificationType.VARIANT for child in children
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


def process_initial_groups(
    db_path: Path, next_group_id=0, iteration_id=0
) -> list[GroupCategoryEntry]:
    """
    Process the initial groupings for the partial categories
    """
    with get_session(db_path) as session:
        # partial_categories = session.query(PartialNameCategory).all()
        category_folder_pair = (
            session.query(PartialNameCategory, Folder)
            .join(Folder, PartialNameCategory.folder_id == Folder.id)
            .filter(PartialNameCategory.classification != ClassificationType.VARIANT)
            .all()
        )

        group_entry_map = defaultdict(list)
        for category, folder in category_folder_pair:
            group_entry_map[category.name].append(category)

        for group_name, category_entries in group_entry_map.items():
            group_category = GroupCategory(
                name=group_name,
                count=len(category_entries),
                group_confidence=1,
                needs_review=False,
                iteration_id=iteration_id,
                id=next_group_id,
            )
            session.add(group_category)
            for category in category_entries:
                entry = GroupCategoryEntry(
                    folder_id=category.folder_id,
                    partial_category_id=category.id,
                    group_id=next_group_id,
                    iteration_id=iteration_id,
                    pre_processed_name=category.original_name,
                    processed_name=group_name,
                    path=str(folder.folder_path),
                    confidence=0.8,
                    processed=False,
                )
                session.add(entry)
            next_group_id += 1

        session.commit()


def process_folders_to_groups(session, group_id: int, iteration_id: int) -> GroupCategoryEntry:
    """
    Process the folders to groups
    """
    folders = session.query(Folder).all()
    for folder in folders:
        group_entry = GroupCategoryEntry(
            folder_id=folder.id,
            group_id=group_id,
            iteration_id=iteration_id,
            pre_processed_name=folder.folder_name,
            processed_name=folder.cleaned_name,
            path=str(folder.folder_path),
            confidence=1.0,
            processed=False,
        )
        session.add(group_entry)
    session.commit()

def refine_groups(session,
    current_group_categories: list[GroupCategoryEntry], iteration_id, next_group_id=1
) -> list[GroupCategory]:
    """
    After groupings have been calculated, evalute the groups to determine which represent
    "real" groupings (i.e. single unique name, or multiple names with a common prefix), and
    which are unable to be grouped.

    """
    print(f"Refining groups, starting at {next_group_id=}")
    clusters: dict[str, list[GroupCategoryEntry]] = defaultdict(list)
    for entry in current_group_categories:
        clusters[entry.cluster_id].append(entry)

    # group_categories: list[GroupCategory] = []
    current_group_id = next_group_id

    for cluster_id, cluster_entries in clusters.items():
        # singletons - just store them and move on
        if len(cluster_entries) == 1:
            entry = cluster_entries[0]

            group_category = GroupCategory(
                id=current_group_id,
                name=entry.pre_processed_name,
                count=1,
                group_confidence=1,
                iteration_id=iteration_id,
                needs_review=False,
            )

            entry.group_id = current_group_id
            entry.processed_name = entry.pre_processed_name
            entry.processed = True

            # group_categories.append(group_category)
            session.add(group_category)
            session.flush()
            current_group_id += 1
            
            continue

        record_names = [record.pre_processed_name for record in cluster_entries]

        # evaluate the group - this does a few things
        # 1. try to normalize the names to a common prefix correcting for spelling
        # 2. determine sub-categories when the group name is a common prefix
        group_name_to_groups = refine_group(record_names) 

        # Create groups based on refined results
        for group_name, entries_by_name in group_name_to_groups.items():
            subcategories = list(set([
                entry.categories[1]
                for entry in entries_by_name
                if len(entry.categories) > 1
            ]))
            member_confidences = []

            group_category = GroupCategory(
                id=current_group_id,
                name=group_name,
                iteration_id=iteration_id,
            )
            subgroups = {
                subcategory: GroupCategory(id=sub_id, name=subcategory, iteration_id=iteration_id)
                for sub_id, subcategory in enumerate(
                    subcategories, start=current_group_id+1
                )
            }
            subcategory_to_entry = {
                subcategory: []
                for subcategory in subcategories
            }

            # Find entries that match this refined group
            group_members = []
            for group_entry in entries_by_name:
                matching_entries = [
                    e
                    for e in cluster_entries
                    if e.pre_processed_name == group_entry.original_name
                ]

                for entry in matching_entries:
                    entry.group_id = group_category.id
                    entry.processed = True
                    entry.confidence = min(entry.confidence, group_entry.confidence)
                    entry.derived_names = group_entry.categories
                    entry.iteration_id = iteration_id

                    entry.processed_name = group_entry.categories[0]

                    member_confidences.append(entry.confidence)
                    group_members.append(entry)

                    if len(group_entry.categories) > 1:
                        subgroup = group_entry.categories[1]
                        subgroup_entry = GroupCategoryEntry(
                            folder_id=entry.folder_id,
                            partial_category_id=entry.partial_category_id,
                            group_id=subgroups[subgroup].id,
                            pre_processed_name=entry.pre_processed_name,
                            processed_name=subgroup,
                            path=entry.path,
                            confidence=entry.confidence,
                            processed=True,
                            iteration_id=iteration_id,
                            derived_names=entry.derived_names,
                        )
                        session.add(subgroup_entry)
                        subcategory_to_entry[subgroup].append(subgroup_entry)
            session.flush()

            current_group_id += len(subgroups) + 1


            # Calculate overall group confidence
            group_category.overall_confidence = min(member_confidences) if member_confidences else 0.5
            group_category.count = len(group_members)
            group_category.needs_review = group_category.overall_confidence < REVIEW_CONFIDENCE_THRESHOLD

            # group_categories.append(group_category)
            session.add(group_category)

            for _, subgroup in subgroups.items():
                sub_member_confidences = [
                    e.confidence
                    for e in subcategory_to_entry[subgroup.name]
                ]
                subgroup.overall_confidence = min(sub_member_confidences) if sub_member_confidences else 0.5
                subgroup.count = len([e for e in group_members if e.processed_name == subgroup.name])
                subgroup.needs_review = subgroup.overall_confidence < REVIEW_CONFIDENCE_THRESHOLD

                # group_categories.append(subgroup)
                session.add(subgroup)
            session.flush()

        session.flush()


    # id_to_group = defaultdict(list)
    # for group in group_categories:
    #     id_to_group[group.id].append(group)
    # duplicate_ids = [id for id, groups in id_to_group.items() if len(groups) > 1]
    # print(f"Duplicate group ids: {duplicate_ids}")
    session.commit()
    # return group_categories


def group_iteration(session: Session, iteration_id: int) -> None:
    """
    Run a single iteration of the grouping process
    """
    # Get last round's entries
    uncertain_categories: tuple[GroupCategoryEntry, Folder] = (
        session.query(GroupCategoryEntry, Folder)
        .join(Folder, GroupCategoryEntry.folder_id == Folder.id, isouter=True)
        .filter(GroupCategoryEntry.iteration_id == iteration_id - 1)
        .all()
    )

    items = prepare_records(uncertain_categories)
    if iteration_id == 1:
        text_distance_ratio = TEXT_DISTANCE_RATIO
    else:
        text_distance_ratio = 0.9

    group_category_entries = cluster_with_custom_metric(
        items, iteration_id=iteration_id, text_distance_ratio=text_distance_ratio
    )
    session.add_all(group_category_entries)
    session.commit()

    next_group_id = session.query(GroupCategory).count() + 1

    refine_groups(session, group_category_entries, iteration_id, next_group_id)
    # session.add_all(groups)
    # session.commit()


def group_folders(db_path: Path, max_iterations: int = 2, review_callback=None) -> None:
    """
    Grouping steps:
        1. Using NLP heuristics, cluster the records in PartialNameCategory to find category names which should be grouped together
            * This populates the GroupCategoryEntry table with the name (matching to PartialNameCategory)
                and the id of the group it has been clustered into
        2. Evaluate the clusters to determine which ones represent a genuine match
            * this creates sub-clusters where all entries in the cluster start with the same string
            * Calculated groups are stored in GroupCategory, and the confidence is set to the
                lowest confidence score of the grouped entries


    """

    # setup the database
    setup_folder_categories(db_path)
    # setup_category_summarization(db_path)
    setup_group(db_path)

    sessionmaker = get_sessionmaker(db_path)
    with sessionmaker() as session:
        process_folders_to_groups(session, 0, 0)  

        for i in range(0, max_iterations):
            # Get groups needing review
            group_iteration(session, i + 1)
            # if review_callback:
            #     review_callback(i)
            print(f"Completed iteration {i+1}")

    """
    Possible next steps:
        * Re-group the categories to find overlaps that can be collapsed further
        * Within categories, find sub-categories that can be collapsed
    """
