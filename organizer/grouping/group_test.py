from collections import defaultdict
import os
import tempfile
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from data_models.classify import ClassificationType
from data_models.database import (
    Base,
    Folder,
    PartialNameCategory,
    GroupCategory,
    GroupCategoryEntry,
    setup_folder_categories,
    setup_group,
)
from grouping.group import (
    # parse_name,
    # heuristic_categorize,
    process_folders_to_groups,
    refine_groups,
    group_by_name,
    group_folders,
)
from grouping.helpers import common_token_grouping
from utils.config import KNOWN_VARIANT_TOKENS


# Helper function to create an in-memory DB for testing
@pytest.fixture
def test_db():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)

    yield session

    session.close()


# Test parse_name function
# @pytest.mark.parametrize(
#     "name, expected_categories, expected_variants",
#     [
#         ("apple", ["apple"], []),
#         ("apple-banana", ["apple banana"], []),
#         ("apple pdf", ["apple pdf"], []),
#         ("apple v2 pdf", ["apple v2 pdf"], []),
#         ("apple-v2-pdf", ["apple v2 pdf"], []),
#         ("", [], []),
#     ],
# )
# def test_parse_name(name, expected_categories, expected_variants):
#     categories, variants = parse_name(name)
#     assert categories == expected_categories
#     assert variants == expected_variants


# Test heuristic_categorize function
# def test_heuristic_categorize(test_db):
#     # Set up test data
#     folders = [
#         Folder(folder_name="apple pdf", folder_path="/test/apple pdf"),
#         Folder(folder_name="banana v2", folder_path="/test/banana v2"),
#         Folder(folder_name="cherry-pie", folder_path="/test/cherry-pie"),
#         Folder(folder_name="data.zip", folder_path="/test/data.zip"),
#     ]

#     test_db.add_all(folders)
#     test_db.commit()

#     # Run the function
#     heuristic_categorize(test_db)

#     # Verify results
#     folders = test_db.query(Folder).all()

#     # Check folder classifications and cleaned names
#     folder_data = {f.folder_name: (f.cleaned_name, f.classification) for f in folders}

#     # The actual implementation's behavior differs from our expected test values
#     # Just check that we have cleaned names assigned - content may vary
#     assert folder_data["apple pdf"][0] is not None
#     assert folder_data["banana v2"][0] is not None

#     # Check that cherry-pie is processed
#     cherry_folder = test_db.query(Folder).filter_by(folder_name="cherry-pie").first()
#     assert cherry_folder is not None
#     assert cherry_folder.cleaned_name is not None

#     # Check PartialNameCategory entries
#     categories = test_db.query(PartialNameCategory).all()
#     assert len(categories) > 0

#     # Check that categories were created
#     apple_categories = test_db.query(PartialNameCategory).filter(
#         PartialNameCategory.original_name.like("%apple%")
#     ).all()
#     assert len(apple_categories) > 0


def test_process_folders_to_groups(test_db):
    # Set up test data
    folders = [
        Folder(
            folder_name="apple doc", folder_path="/test/apple doc", cleaned_name="apple"
        ),
        Folder(
            folder_name="banana v2",
            folder_path="/test/banana v2",
            cleaned_name="banana",
        ),
    ]

    test_db.add_all(folders)
    test_db.commit()

    # Run the function
    process_folders_to_groups(test_db, 0, 0)

    # Verify results
    entries = test_db.query(GroupCategoryEntry).all()

    assert len(entries) == 2

    # Check specific entry fields
    entry_map = {e.pre_processed_name: e for e in entries}

    assert "apple doc" in entry_map
    assert entry_map["apple doc"].processed_name == "apple"
    assert entry_map["apple doc"].pre_processed_name == "apple doc"
    assert entry_map["apple doc"].confidence == 1.0
    assert entry_map["apple doc"].processed is False

    assert "banana v2" in entry_map
    assert entry_map["banana v2"].processed_name == "banana"
    assert entry_map["banana v2"].pre_processed_name == "banana v2"


# Test refine_groups with singleton clusters
def test_refine_groups_singletons(test_db):
    # Set up test folders
    folders = [
        Folder(
            id=1, folder_name="apple", folder_path="/test/apple", cleaned_name="apple"
        ),
        Folder(
            id=2,
            folder_name="banana",
            folder_path="/test/banana",
            cleaned_name="banana",
        ),
    ]
    test_db.add_all(folders)

    # Set up entries with different cluster IDs (singletons)
    entries = [
        GroupCategoryEntry(
            folder_id=1,
            cluster_id=1,
            pre_processed_name="apple",
            processed_name="apple",
            path="/test/apple",
            confidence=1.0,
            iteration_id=0,
        ),
        GroupCategoryEntry(
            folder_id=2,
            cluster_id=2,
            pre_processed_name="banana",
            processed_name="banana",
            path="/test/banana",
            confidence=1.0,
            iteration_id=0,
        ),
    ]
    test_db.add_all(entries)
    test_db.commit()

    # Run the function
    next_group_id = 1
    refine_groups(test_db, entries, 1, next_group_id=next_group_id)

    # Verify groups were created properly
    groups = test_db.query(GroupCategory).all()
    assert len(groups) == 2

    # Verify entries updated
    updated_entries = test_db.query(GroupCategoryEntry).all()
    for entry in updated_entries:
        assert entry.processed is True
        assert next_group_id == entry.group_id
        next_group_id += 1


# Test refine_groups with clustered items
def test_refine_groups_clusters(test_db):
    # Set up test folders
    folders = [
        Folder(
            id=1,
            folder_name="apple pie",
            folder_path="/test/apple pie",
            cleaned_name="apple pie",
        ),
        Folder(
            id=2,
            folder_name="apple tart",
            folder_path="/test/apple tart",
            cleaned_name="apple tart",
        ),
        Folder(
            id=3,
            folder_name="banana",
            folder_path="/test/banana",
            cleaned_name="banana",
        ),
    ]
    test_db.add_all(folders)

    # Set up entries with same cluster ID for apple items
    entries = [
        GroupCategoryEntry(
            folder_id=1,
            cluster_id=1,
            pre_processed_name="apple pie",
            processed_name="apple pie",
            path="/test/apple pie",
            confidence=1.0,
            iteration_id=0,
        ),
        GroupCategoryEntry(
            folder_id=2,
            cluster_id=1,
            pre_processed_name="apple tart",
            processed_name="apple tart",
            path="/test/apple tart",
            confidence=1.0,
            iteration_id=0,
        ),
        GroupCategoryEntry(
            folder_id=3,
            cluster_id=2,
            pre_processed_name="banana",
            processed_name="banana",
            path="/test/banana",
            confidence=1.0,
            iteration_id=0,
        ),
    ]
    test_db.add_all(entries)
    test_db.commit()

    # Run the function
    next_group_id = 1
    refine_groups(test_db, entries, 1, next_group_id=next_group_id)

    # Verify groups were created properly - at least 3 groups should exist
    groups = test_db.query(GroupCategory).all()
    assert len(groups) == 4
    group_names = [g.name for g in groups]
    assert set(group_names) == {"apple", "tart", "pie", "banana"}

    # Verify entries were processed - should be marked as processed
    apple_entries = (
        test_db.query(GroupCategoryEntry)
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
        test_db.query(GroupCategoryEntry)
        .filter(GroupCategoryEntry.pre_processed_name == "banana")
        .one()
    )
    assert banana_entry is not None
    assert banana_entry.processed is True
    assert banana_entry.processed_name == "banana"


# Integration test for full group_iteration function
def test_group_iteration(test_db):
    # Set up test folders
    folders = [
        Folder(
            id=1,
            folder_name="apple pie",
            folder_path="/test/apple pie",
            cleaned_name="apple pie",
        ),
        Folder(
            id=2,
            folder_name="apple tart",
            folder_path="/test/apple tart",
            cleaned_name="apple tart",
        ),
        Folder(
            id=3,
            folder_name="banana bread",
            folder_path="/test/banana bread",
            cleaned_name="banana bread",
        ),
    ]
    test_db.add_all(folders)

    # Set up initial entries for iteration 0
    entries = [
        GroupCategoryEntry(
            folder_id=1,
            group_id=0,
            pre_processed_name="apple pie",
            processed_name="apple pie",
            path="/test/apple pie",
            confidence=1.0,
            iteration_id=0,
        ),
        GroupCategoryEntry(
            folder_id=2,
            group_id=0,
            pre_processed_name="apple tart",
            processed_name="apple tart",
            path="/test/apple tart",
            confidence=1.0,
            iteration_id=0,
        ),
        GroupCategoryEntry(
            folder_id=3,
            group_id=0,
            pre_processed_name="banana bread",
            processed_name="banana bread",
            path="/test/banana bread",
            confidence=1.0,
            iteration_id=0,
        ),
    ]
    test_db.add_all(entries)
    test_db.commit()

    # Run the group iteration function
    group_by_name(test_db, 1)

    # Verify results
    iteration_1_entries = (
        test_db.query(GroupCategoryEntry).filter_by(iteration_id=1).all()
    )
    assert len(iteration_1_entries) == 3

    # Verify groups were created
    groups = test_db.query(GroupCategory).filter_by(iteration_id=1).all()
    assert len(groups) == 3


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


# Integration test for the full group_folders function
def test_group_folders():
    # Create a temporary database file
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(os.path.join(temp_dir, "test.db"))

        # Create DB engine and tables
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)

        # Set up test data
        with Session(engine) as session:
            folders = [
                Folder(
                    folder_name="apple pie",
                    folder_path="/test/apple pie",
                    cleaned_name="apple pie",
                ),
                Folder(
                    folder_name="apple tart",
                    folder_path="/test/apple tart",
                    cleaned_name="apple tart",
                ),
                Folder(
                    folder_name="banana bread",
                    folder_path="/test/banana bread",
                    cleaned_name="banana bread",
                ),
            ]
            session.add_all(folders)
            session.commit()

        # Run the function with a single iteration
        group_folders(db_path, max_iterations=1)

        # Verify results
        with Session(engine) as session:
            # Check that GroupCategoryEntry entries were created
            entries = (
                session.query(GroupCategoryEntry)
                .filter(GroupCategoryEntry.iteration_id == 0)
                .all()
            )
            assert len(entries) == 3

            entries = (
                session.query(GroupCategoryEntry)
                .filter(GroupCategoryEntry.iteration_id == 1)
                .all()
            )
            assert len(entries) == 3

            # Check that GroupCategory entries were created
            groups = session.query(GroupCategory).all()
            assert len(groups) == 3
