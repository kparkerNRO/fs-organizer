import pytest
from storage.factories import (
    NodeFactory,
)
from storage.manager import NodeKind
from storage.work_models import GroupCategory, GroupCategoryEntry
from utils.config import get_minimal_config

from stages.grouping.group import (
    group_folders,
    _process_folders_to_groups,
)
from stages.grouping.helpers import common_token_grouping


def test_process_folders_to_groups(
    index_session, work_session, sample_run, sample_snapshot
):
    # Set up test data
    NodeFactory(snapshot_id=sample_snapshot.id, name="apple doc", kind=NodeKind.DIR)
    NodeFactory(snapshot_id=sample_snapshot.id, name="banana v2", kind=NodeKind.DIR)

    # Run the function (it will create iteration 0)
    # Pass None for group_id since we don't have a group yet
    _process_folders_to_groups(
        index_session=index_session,
        work_session=work_session,
        run_id=sample_run.id,
        snapshot_id=sample_snapshot.id,
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
                "A wild test one": ["A wild test", "one"],
                "A wild test three": ["A wild test", "three"],
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
                "A wild test one": ["A wild test", "one"],
                "A wild test three": ["A wild test", "three"],
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
):
    NodeFactory(
        snapshot_id=storage_snapshot.id,
        name="apple pie",
        kind=NodeKind.DIR,
    )
    NodeFactory(
        snapshot_id=storage_snapshot.id,
        name="apple tart",
        kind=NodeKind.DIR,
    )
    NodeFactory(
        snapshot_id=storage_snapshot.id,
        name="banana bread",
        kind=NodeKind.DIR,
    )
    storage_index_session.commit()

    group_folders(
        storage_manager,
        max_iterations=1,
        config=get_minimal_config(),
        snapshot_id=storage_snapshot.id,
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

    # Iteration 3: Folder name grouping (new step)
    # This step identifies "apple" as a common prefix and creates:
    # - "apple" and "pie" for folder "apple pie"
    # - "apple" and "tart" for folder "apple tart"
    # - "banana bread" (no grouping)
    entries_iter3 = (
        storage_work_session.query(GroupCategoryEntry)
        .filter(GroupCategoryEntry.iteration_id == 3)
        .all()
    )
    assert len(entries_iter3) == 5  # apple (2x), pie, tart, banana bread

    # After compacting and creating exact groups, we should have 4 groups:
    # "apple" (2 folders), "pie" (1 folder), "tart" (1 folder), "banana bread" (1 folder)
    groups = storage_work_session.query(GroupCategory).all()
    assert len(groups) == 4

    # Verify the groups
    group_map = {g.name: g for g in groups}
    assert "apple" in group_map
    assert group_map["apple"].count == 2  # Both apple pie and apple tart
    assert "pie" in group_map
    assert group_map["pie"].count == 1
    assert "tart" in group_map
    assert group_map["tart"].count == 1
    assert "banana bread" in group_map
    assert group_map["banana bread"].count == 1
