from collections import defaultdict
from datetime import datetime
from logging import getLogger

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from stages.grouping.helpers import get_next_iteration_id
from storage.id_defaults import get_effective_snapshot_id
from storage.index_models import Node
from storage.manager import NodeKind, StorageManager
from storage.work_models import (
    GroupCategory,
    GroupCategoryEntry,
    GroupIteration,
    Run,
)
from utils.config import Config, get_config
from utils.filename_processing import clean_filename, split_view_type

from stages.grouping.tag_decomposition import decompose_compound_tags
from stages.grouping.folder_name_grouping import apply_folder_name_grouping
from stages.grouping.group_cleanup import apply_group_cleanup

REVIEW_CONFIDENCE_THRESHOLD = 0.7
logger = getLogger(__name__)


def _process_folders_to_groups(
    index_session, work_session, run_id: int, snapshot_id: int
):
    """
    Process the folders to groups
    """
    logger.info("Grouping folders")
    iteration_id = get_next_iteration_id(work_session)

    # Create the iteration record with required fields
    iteration = GroupIteration(
        id=iteration_id,
        run_id=run_id,
        snapshot_id=snapshot_id,
        description="Initial folder processing",
    )
    work_session.add(iteration)
    work_session.flush()

    query = (
        select(Node)
        .where(Node.snapshot_id == snapshot_id)
        .where(Node.kind == NodeKind.DIR)
    )
    folders = index_session.execute(query).scalars().all()
    logger.info(f"Found {len(folders)} folders")

    for folder in folders:
        # Clean the folder name for processing
        cleaned_name = clean_filename(folder.name)

        group_entry = GroupCategoryEntry(
            folder_id=folder.id,
            iteration_id=iteration_id,
            pre_processed_name=folder.name,
            processed_name=cleaned_name,
            path=str(folder.abs_path),
            confidence=1.0,
            processed=False,
        )
        work_session.add(group_entry)
    work_session.commit()

    logger.info("Groups created")


def _pre_process_groups(
    session: Session,
    config: Config,
    run_id: int,
    snapshot_id: int,
) -> None:
    """
    Clean up compound entries - split out hyphen-delineated values
    """
    logger.info("pre-processing groups")
    # Get last round's entries
    iteration_id = get_next_iteration_id(session)

    # Create the iteration record with required fields
    iteration = GroupIteration(
        id=iteration_id,
        run_id=run_id,
        snapshot_id=snapshot_id,
        description="Pre-processing groups (compound tag splitting)",
    )
    session.add(iteration)
    session.flush()

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

    for category in uncertain_categories:
        # Use processed_name if available, otherwise fall back to pre_processed_name
        name_to_process = category["processed_name"] or category["pre_processed_name"]
        if not name_to_process:
            continue  # Skip entries with no name
        split_name = name_to_process.split("-")  # type: ignore[union-attr]
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

    logger.info("Pre-processing complete")
    session.commit()


def _compact_groups(
    index_session: Session,
    work_session: Session,
    run_id: int,
    snapshot_id: int,
):
    """
    Compact groups for each folder, making sure that folders don't have duplicate category entries
    """

    iteration_id = get_next_iteration_id(work_session)

    # Create the iteration record
    iteration = GroupIteration(
        id=iteration_id,
        run_id=run_id,
        snapshot_id=snapshot_id,
        description="Compact groups",
    )
    work_session.add(iteration)
    work_session.commit()

    folders = (
        index_session.execute(
            select(Node)
            .where(Node.snapshot_id == snapshot_id)
            .where(Node.kind == NodeKind.DIR)
        )
        .scalars()
        .all()
    )
    for folder in folders:
        groups = (
            work_session.execute(
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
                    if existing_group.pre_processed_name and group.pre_processed_name:
                        existing_group.pre_processed_name = (
                            existing_group.pre_processed_name
                            + ";"
                            + group.pre_processed_name
                        )

            else:
                new_entry = GroupCategoryEntry(
                    folder_id=folder.id,
                    processed_name=group.processed_name,
                    pre_processed_name=group.pre_processed_name,
                    derived_names=group.derived_names,
                    path=group.path,
                    confidence=group.confidence,
                    iteration_id=iteration_id,
                )
                work_session.add(new_entry)
                new_group_name_map[processed_name] = new_entry  # type: ignore[index]  # ty bug: processed_name is str at runtime
    work_session.commit()


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
    session_manager: StorageManager,
    max_iterations: int = 2,
    review_callback=None,
    config: Config | None = None,
    snapshot_id: int | None = None,
) -> int:
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
    config = config or get_config()

    # setup the database
    logger.info("Beginning grouping process")
    with (
        session_manager.get_work_session() as work_session,
        session_manager.get_index_session() as index_session,
    ):
        snapshot_id = get_effective_snapshot_id(session_manager, snapshot_id)

        # Get run_id and snapshot_id if not provided
        run = Run(snapshot_id=snapshot_id, started_at=datetime.now(), status="running")
        work_session.add(run)
        work_session.flush()

        logger.info(f"Using {snapshot_id=}, {run.id=}")

        _process_folders_to_groups(
            work_session=work_session,
            index_session=index_session,
            run_id=run.id,
            snapshot_id=snapshot_id,
        )
        _pre_process_groups(
            work_session, config=config, run_id=run.id, snapshot_id=snapshot_id
        )
        work_session.commit()

        # decompose_compound_tags(work_session, run_id=run.id, snapshot_id=snapshot_id)

        # too agressive
        apply_folder_name_grouping(work_session, run_id=run.id, snapshot_id=snapshot_id)

        # not aggressive enough...
        apply_group_cleanup(work_session, run_id=run.id, snapshot_id=snapshot_id)
        

        _compact_groups(
            work_session=work_session,
            index_session=index_session,
            run_id=run.id,
            snapshot_id=snapshot_id,
        )
        _create_exact_groups(work_session)
        return run.id
