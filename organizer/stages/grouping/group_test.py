import pytest
from typing import cast
from storage.factories import (
    GroupCategoryEntryFactory,
    GroupIterationFactory,
    NodeFactory,
)
from storage.manager import NodeKind
from storage.work_models import GroupCategory, GroupCategoryEntry, GroupIteration

from stages.grouping.group import (
    group_folders,
    process_folders_to_groups,
    refine_groups,
)
from stages.grouping.helpers import common_token_grouping
from utils.config import get_minimal_config


def test_process_folders_to_groups(
    index_session, work_session, sample_run, sample_snapshot
):
    # Set up test data
    NodeFactory(
        snapshot_id=sample_snapshot.snapshot_id, name="apple doc", kind=NodeKind.DIR
    )
    NodeFactory(
        snapshot_id=sample_snapshot.snapshot_id, name="banana v2", kind=NodeKind.DIR
    )

    # Run the function (it will create iteration 0)
    # Pass None for group_id since we don't have a group yet
    process_folders_to_groups(
        index_session=index_session,
        work_session=work_session,
        group_id=None,
        run_id=sample_run.id,
        snapshot_id=sample_snapshot.snapshot_id,
    )

    # Verify results
    entries = work_session.query(GroupCategoryEntry).all()

    assert len(entries) == 2

    # Check specific entry fields
    entry_map = {e.pre_processed_name: e for e in entries}

    assert "apple doc" in entry_map
    assert entry_map["apple doc"].processed_name == "apple doc"
    assert entry_map["apple doc"].pre_processed_name == "apple doc"
    assert entry_map["apple doc"].confidence == 1.0
    assert entry_map["apple doc"].processed is False

    assert "banana v2" in entry_map
    # Note: clean_filename removes trailing numbers, so "v2" becomes "v"
    assert entry_map["banana v2"].processed_name == "banana v"
    assert entry_map["banana v2"].pre_processed_name == "banana v2"


# Test refine_groups with singleton clusters
def test_refine_groups_singletons(
    index_session, work_session, sample_run, sample_snapshot
):
    # Create iteration records
    iteration0 = cast(
        GroupIteration,
        GroupIterationFactory(
            id=0,
            description="test iteration 0",
            run=sample_run,
        ),
    )
    GroupIterationFactory(
        id=1,
        description="test iteration 1",
        run=sample_run,
    )

    # Set up entries with different cluster IDs (singletons)
    entries: list[GroupCategoryEntry] = [
        cast(
            GroupCategoryEntry,
            GroupCategoryEntryFactory(
                folder_id=1,
                cluster_id=1,
                pre_processed_name="apple",
                processed_name="apple",
                path="/test/apple",
                confidence=1.0,
                iteration_id=iteration0.id,
            ),
        ),
        cast(
            GroupCategoryEntry,
            GroupCategoryEntryFactory(
                folder_id=2,
                cluster_id=2,
                pre_processed_name="banana",
                processed_name="banana",
                path="/test/banana",
                confidence=1.0,
                iteration_id=iteration0.id,
            ),
        ),
    ]

    # Run the function
    next_group_id = 1
    refine_groups(work_session, entries, 1, next_group_id=next_group_id)

    # Verify groups were created properly
    groups = work_session.query(GroupCategory).all()
    assert len(groups) == 2

    # Verify entries updated
    updated_entries = work_session.query(GroupCategoryEntry).all()
    for entry in updated_entries:
        assert entry.processed is True
        assert next_group_id == entry.group_id
        next_group_id += 1


# Test refine_groups with clustered items
def test_refine_groups_clusters(
    index_session, work_session, sample_run, sample_snapshot
):
    # Create iteration records
    iteration0 = cast(
        GroupIteration,
        GroupIterationFactory(
            id=0,
            description="test iteration 0",
            run=sample_run,
        ),
    )
    GroupIterationFactory(
        id=1,
        description="test iteration 1",
        run=sample_run,
    )

    # Set up entries with same cluster ID for apple items
    entries: list[GroupCategoryEntry] = [
        cast(
            GroupCategoryEntry,
            GroupCategoryEntryFactory(
                folder_id=1,
                cluster_id=1,
                pre_processed_name="apple pie",
                processed_name="apple pie",
                path="/test/apple pie",
                confidence=1.0,
                iteration_id=iteration0.id,
            ),
        ),
        cast(
            GroupCategoryEntry,
            GroupCategoryEntryFactory(
                folder_id=2,
                cluster_id=1,
                pre_processed_name="apple tart",
                processed_name="apple tart",
                path="/test/apple tart",
                confidence=1.0,
                iteration_id=iteration0.id,
            ),
        ),
        cast(
            GroupCategoryEntry,
            GroupCategoryEntryFactory(
                folder_id=3,
                cluster_id=2,
                pre_processed_name="banana",
                processed_name="banana",
                path="/test/banana",
                confidence=1.0,
                iteration_id=iteration0.id,
            ),
        ),
    ]

    # Run the function
    next_group_id = 1
    refine_groups(work_session, entries, 1, next_group_id=next_group_id)

    # Verify groups were created properly - at least 3 groups should exist
    groups = work_session.query(GroupCategory).all()
    assert len(groups) == 4
    group_names = [g.name for g in groups]
    assert set(group_names) == {"apple", "tart", "pie", "banana"}

    # Verify entries were processed - should be marked as processed
    apple_entries = (
        work_session.query(GroupCategoryEntry)
        .filter(GroupCategoryEntry.pre_processed_name.in_(["apple pie", "apple tart"]))
        .all()
    )

    assert len(apple_entries) == 4
    categories = [e.processed_name for e in apple_entries]
    assert set(categories) == {"apple", "tart", "pie"}
    for entry in apple_entries:
        assert entry.processed is True

    # Check banana entry is processed
    banana_entry = (
        work_session.query(GroupCategoryEntry)
        .filter(GroupCategoryEntry.pre_processed_name == "banana")
        .one()
    )
    assert banana_entry is not None
    assert banana_entry.processed is True
    assert banana_entry.processed_name == "banana"


# Test common_token_grouping for integration with refine_groups
@pytest.mark.parametrize(
    "name, input_list, expected",
    [
        ("empty_list", [], None),
        ("single_name", ["example"], {}),
        (
            "multi_word_overlap",
            ["mom's apple pie", "mom's apple tart"],
            {
                "mom's apple pie": ["mom's apple", "pie"],
                "mom's apple tart": ["mom's apple", "tart"],
            },
        ),
        ("no_common_tokens", ["apple", "banana", "cherry"], {}),
        (
            "common_prefix",
            ["apple pie", "apple tart", "apple juice"],
            {
                "apple pie": ["apple", "pie"],
                "apple tart": ["apple", "tart"],
                "apple juice": ["apple", "juice"],
            },
        ),
        (
            "common_suffix",
            ["pie apple", "tart apple", "juice apple"],
            {},
        ),
        (
            "mixed_case",
            ["Apple Pie", "apple tart", "APPLE juice"],
            {
                "Apple Pie": ["Apple", "Pie"],
                "apple tart": ["apple", "tart"],
                "APPLE juice": ["apple", "juice"],
            },
        ),
        (
            "numbers_in_names",
            ["apple 1", "apple 2", "apple 3"],
            {},
        ),
        (
            "partial_overlap",
            ["apple pie", "apple tart", "banana pie"],
            {"apple pie": ["apple", "pie"], "apple tart": ["apple", "tart"]},
        ),
        (
            "overlapping_groups",
            [
                "A wild two",
                "A wild four",
                "A wild test one",
                "A wild test three",
            ],
            {
                "A wild test one": ["A wild", "test one"],
                "A wild test three": ["A wild", "test three"],
                "A wild two": ["A wild", "two"],
                "A wild four": ["A wild", "four"],
            },
        ),
        (
            # order matters in how it processes the groups, so
            # test both ordering
            "overlapping_groups_inverted",
            [
                "A wild test one",
                "A wild test three",
                "A wild two",
                "A wild four",
            ],
            {
                "A wild test one": ["A wild", "test one"],
                "A wild test three": ["A wild", "test three"],
                "A wild two": ["A wild", "two"],
                "A wild four": ["A wild", "four"],
            },
        ),
        (
            "5e no match",
            ["Dagons Deliverance 5e", "Dagons Deliverance"],
            {
                "Dagons Deliverance 5e": ["Dagons Deliverance", "5e"],
                "Dagons Deliverance": ["Dagons Deliverance"],
            },
        ),
    ],
)
def test_common_token_grouping(name, input_list, expected):
    result = common_token_grouping(input_list)
    assert result == expected


def test_group_folders(
    storage_manager,
    storage_index_session,
    storage_work_session,
    storage_snapshot,
    storage_run,
):
    NodeFactory(
        snapshot_id=storage_snapshot.snapshot_id,
        name="apple pie",
        kind=NodeKind.DIR,
    )
    NodeFactory(
        snapshot_id=storage_snapshot.snapshot_id,
        name="apple tart",
        kind=NodeKind.DIR,
    )
    NodeFactory(
        snapshot_id=storage_snapshot.snapshot_id,
        name="banana bread",
        kind=NodeKind.DIR,
    )
    storage_index_session.commit()

    group_folders(
        storage_manager,
        max_iterations=1,
        config=get_minimal_config(),
        run_id=storage_run.id,
        snapshot_id=storage_snapshot.snapshot_id,
    )

    entries_iter0 = (
        storage_work_session.query(GroupCategoryEntry)
        .filter(GroupCategoryEntry.iteration_id == 0)
        .all()
    )
    assert len(entries_iter0) == 3

    entries_iter1 = (
        storage_work_session.query(GroupCategoryEntry)
        .filter(GroupCategoryEntry.iteration_id == 1)
        .all()
    )
    assert len(entries_iter1) == 3

    entries_iter2 = (
        storage_work_session.query(GroupCategoryEntry)
        .filter(GroupCategoryEntry.iteration_id == 2)
        .all()
    )
    assert len(entries_iter2) == 3

    groups = storage_work_session.query(GroupCategory).all()
    assert len(groups) == 3
