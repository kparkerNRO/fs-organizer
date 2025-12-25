from collections import defaultdict

from sqlalchemy.orm import Session

from data_models.database import (
    GroupCategory,
    setup_folder_categories,
    setup_group,
    get_sessionmaker,
    Folder,
    GroupCategoryEntry,
)
from grouping.group_cleanup import refine_group
from logging import getLogger
from grouping.nlp_grouping import (
    cluster_with_custom_metric,
    prepare_records,
    compute_custom_distance_matrix,
)
from utils.config import Config, get_config
from utils.filename_utils import clean_filename, split_view_type

from pathlib import Path
from sqlalchemy import func, select

REVIEW_CONFIDENCE_THRESHOLD = 0.7
log = getLogger(__name__)


def get_next_iteration_id(session: Session):
    result = session.execute(
        select(func.max(GroupCategoryEntry.iteration_id))
    ).scalar_one()

    if result is None:
        return 0
    return result + 1


def process_folders_to_groups(session, group_id: int):
    """
    Process the folders to groups
    """
    iteration_id = get_next_iteration_id(session)
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


def refine_groups(
    session,
    current_group_categories: list[GroupCategoryEntry],
    iteration_id,
    next_group_id=1,
) -> int:
    """
    After groupings have been calculated, evalute the groups to determine which represent
    "real" groupings (i.e. single unique name, or multiple names with a common prefix), and
    which are unable to be grouped.

    """
    log.info(f"Refining groups, starting at next_group_id={next_group_id}")
    clusters: dict[str, list[GroupCategoryEntry]] = defaultdict(list)
    for entry in current_group_categories:
        clusters[entry.cluster_id].append(entry)

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
            subcategories = list(
                set(
                    [
                        entry.categories[1]
                        for entry in entries_by_name
                        if entry.categories and len(entry.categories) > 1
                    ]
                )
            )
            member_confidences = []

            group_category = GroupCategory(
                id=current_group_id,
                name=group_name,
                iteration_id=iteration_id,
            )
            subgroups = {
                subcategory: GroupCategory(
                    id=sub_id, name=subcategory, iteration_id=iteration_id
                )
                for sub_id, subcategory in enumerate(
                    subcategories, start=current_group_id + 1
                )
            }
            subcategory_to_entry = {subcategory: [] for subcategory in subcategories}

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

                    if group_entry.categories and len(group_entry.categories) > 0:
                        entry.processed_name = group_entry.categories[0]

                    member_confidences.append(entry.confidence)
                    group_members.append(entry)

                    if group_entry.categories and len(group_entry.categories) > 1:
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
            group_category.overall_confidence = (
                min(member_confidences) if member_confidences else 0.5
            )
            group_category.count = len(group_members)
            group_category.needs_review = (
                group_category.overall_confidence < REVIEW_CONFIDENCE_THRESHOLD
            )

            # group_categories.append(group_category)
            session.add(group_category)

            for _, subgroup in subgroups.items():
                sub_member_confidences = [
                    e.confidence for e in subcategory_to_entry[subgroup.name]
                ]
                subgroup.overall_confidence = (
                    min(sub_member_confidences) if sub_member_confidences else 0.5
                )
                subgroup.count = len(
                    [e for e in group_members if e.processed_name == subgroup.name]
                )
                subgroup.needs_review = (
                    subgroup.overall_confidence < REVIEW_CONFIDENCE_THRESHOLD
                )

                session.add(subgroup)
            session.flush()

        session.flush()

    session.commit()
    return current_group_id


def pre_process_groups(session: Session, config: Config | None = None) -> None:
    """
    Clean up compound entries - split out hyphen-delineated values
    """
    # Get last round's entries
    iteration_id = get_next_iteration_id(session)
    stmt = select(GroupCategoryEntry).where(
        GroupCategoryEntry.iteration_id == iteration_id - 1
    )
    uncertain_entries = session.scalars(stmt).all()
    uncertain_categories = [
        {
            "folder_id": entry.folder_id,
            "group_id": entry.group_id,
            "iteration_id": iteration_id,  # Artificially set the id to this iteration
            "pre_processed_name": entry.pre_processed_name,
            "processed_name": entry.processed_name,
            "path": entry.path,
            "confidence": entry.confidence,
            "processed": entry.processed,
            "derived_names": getattr(entry, "derived_names", None),
            "partial_category_id": getattr(entry, "partial_category_id", None),
        }
        for entry in uncertain_entries
    ]

    config = config or get_config()

    for category in uncertain_categories:
        split_name = category["processed_name"].split("-")  # type: ignore[union-attr]  # ty bug: SQLAlchemy ORM attribute should be str
        categories = []
        for name in split_name:
            cleaned_name = clean_filename(name, config=config)
            category_current = cleaned_name
            while category_current:
                entry, variant = split_view_type(
                    category_current, config.known_variant_tokens
                )
                if entry and entry not in categories:
                    categories.append(entry)
                if variant:
                    categories.append(variant)

                if category_current == entry:
                    break

                category_current = entry

        for entry in categories:
            session.add(GroupCategoryEntry(**category | {"processed_name": entry}))
    session.commit()


def compact_groups(session: Session):
    """
    Compact groups for each folder, making sure that folders don't have duplicate category entries
    """
    iteration_id = get_next_iteration_id(session)
    folders = session.execute(select(Folder)).scalars().all()
    for folder in folders:
        groups = (
            session.execute(
                select(GroupCategoryEntry)
                .where(GroupCategoryEntry.folder_id == folder.id)
                .where(GroupCategoryEntry.iteration_id == iteration_id - 1)
            )
            .scalars()
            .all()
        )
        new_group_name_map: dict[str, GroupCategoryEntry] = {}
        for group in groups:
            processed_name = group.processed_name  # type: ignore[assignment]  # ty bug: SQLAlchemy ORM attribute should be str
            if processed_name in new_group_name_map:
                existing_group = new_group_name_map[processed_name]  # type: ignore[index]  # ty bug: processed_name is str at runtime
                existing_group.confidence = min(
                    existing_group.confidence, group.confidence
                )
                if existing_group.pre_processed_name != group.pre_processed_name:
                    existing_group.pre_processed_name = (
                        existing_group.pre_processed_name
                        + ";"
                        + group.pre_processed_name
                    )

            else:
                new_entry = GroupCategoryEntry(
                    folder_id=folder.id,
                    group_id=group.group_id,
                    processed_name=group.processed_name,
                    pre_processed_name=group.pre_processed_name,
                    derived_names=group.derived_names,
                    path=group.path,
                    confidence=group.confidence,
                    iteration_id=iteration_id,
                )
                session.add(new_entry)
                new_group_name_map[processed_name] = new_entry  # type: ignore[index]  # ty bug: processed_name is str at runtime
    session.commit()


def _create_exact_groups(session: Session) -> None:
    iteration_id = get_next_iteration_id(session) - 1
    if iteration_id < 0:
        return

    entries = (
        session.execute(
            select(GroupCategoryEntry).where(
                GroupCategoryEntry.iteration_id == iteration_id
            )
        )
        .scalars()
        .all()
    )
    if not entries:
        return

    max_group_id = session.execute(select(func.max(GroupCategory.id))).scalar_one()
    current_group_id = (max_group_id or 0) + 1

    grouped_entries: dict[str, list[GroupCategoryEntry]] = defaultdict(list)
    for entry in entries:
        group_name = entry.processed_name or entry.pre_processed_name
        grouped_entries[group_name].append(entry)

    for group_name, group_entries in grouped_entries.items():
        confidences = [entry.confidence for entry in group_entries]
        group_confidence = min(confidences) if confidences else 0.5

        group_category = GroupCategory(
            id=current_group_id,
            name=group_name,
            count=len(group_entries),
            group_confidence=group_confidence,
            iteration_id=iteration_id,
            needs_review=group_confidence < REVIEW_CONFIDENCE_THRESHOLD,
        )
        session.add(group_category)
        for entry in group_entries:
            entry.group_id = current_group_id
            entry.processed_name = group_name
            entry.processed = True

        current_group_id += 1

    session.commit()


def group_folders(
    db_path: Path,
    max_iterations: int = 2,
    review_callback=None,
    config: Config | None = None,
) -> None:
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
    setup_group(db_path)

    sessionmaker = get_sessionmaker(db_path)
    with sessionmaker() as session:
        config = config or get_config()
        process_folders_to_groups(session, 0)
        pre_process_groups(session, config=config)

        # New tag decomposition stage
        from grouping.tag_decomposition import decompose_compound_tags

        decompose_compound_tags(session)

        compact_groups(session)
        _create_exact_groups(session)
