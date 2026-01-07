"""
Tests for folder name grouping functionality.

This module tests the _apply_folder_name_grouping function which identifies
common prefixes in folder names and creates hierarchical group structures.
"""

from storage.factories import NodeFactory
from storage.manager import NodeKind
from storage.work_models import GroupCategoryEntry, GroupIteration

from stages.grouping.folder_name_grouping import (
    apply_folder_name_grouping,
    _is_valid_group_prefix,
)


class TestIsValidGroupPrefix:
    """Test the prefix validation logic"""

    def test_empty_prefix(self):
        assert not _is_valid_group_prefix("")
        assert not _is_valid_group_prefix("   ")

    def test_stopword_only_prefix(self):
        assert not _is_valid_group_prefix("the")
        assert not _is_valid_group_prefix("the a")
        assert not _is_valid_group_prefix("of the")

    def test_valid_prefix_with_stopwords(self):
        assert _is_valid_group_prefix("the wizard tower")
        assert _is_valid_group_prefix("the wizard")
        assert _is_valid_group_prefix("wizard tower")

    def test_valid_prefix_no_stopwords(self):
        assert _is_valid_group_prefix("apple")
        assert _is_valid_group_prefix("apple pie")


class TestApplyFolderNameGrouping:
    """Test the folder name grouping function"""

    def test_wizard_tower_grouping(
        self, index_session, work_session, sample_run, sample_snapshot
    ):
        """
        Test Case 1 from requirements:
        Input: ["the wizard tower upstairs", "the wizard tower downstairs", "the wizard tower outside"]
        Expected: Groups created for "the wizard tower" and individual suffixes
        """
        # Create test folders
        folder1 = NodeFactory(
            snapshot_id=sample_snapshot.id,
            name="the wizard tower upstairs",
            kind=NodeKind.DIR,
        )
        folder2 = NodeFactory(
            snapshot_id=sample_snapshot.id,
            name="the wizard tower downstairs",
            kind=NodeKind.DIR,
        )
        folder3 = NodeFactory(
            snapshot_id=sample_snapshot.id,
            name="the wizard tower outside",
            kind=NodeKind.DIR,
        )
        index_session.commit()

        # Create iteration 0 (simulating previous pipeline step)
        iteration = GroupIteration(
            id=0,
            run_id=sample_run.id,
            snapshot_id=sample_snapshot.id,
            description="Test iteration",
        )
        work_session.add(iteration)
        work_session.flush()  # Flush to satisfy foreign key constraint

        # Create initial GroupCategoryEntry records
        entries = [
            GroupCategoryEntry(
                folder_id=folder1.id,
                iteration_id=0,
                pre_processed_name="the wizard tower upstairs",
                processed_name="the wizard tower upstairs",
                path=str(folder1.abs_path),
                confidence=1.0,
                processed=False,
            ),
            GroupCategoryEntry(
                folder_id=folder2.id,
                iteration_id=0,
                pre_processed_name="the wizard tower downstairs",
                processed_name="the wizard tower downstairs",
                path=str(folder2.abs_path),
                confidence=1.0,
                processed=False,
            ),
            GroupCategoryEntry(
                folder_id=folder3.id,
                iteration_id=0,
                pre_processed_name="the wizard tower outside",
                processed_name="the wizard tower outside",
                path=str(folder3.abs_path),
                confidence=1.0,
                processed=False,
            ),
        ]
        for entry in entries:
            work_session.add(entry)
        work_session.commit()

        # Apply folder name grouping
        apply_folder_name_grouping(
            work_session, run_id=sample_run.id, snapshot_id=sample_snapshot.id
        )

        # Verify results
        new_entries = (
            work_session.query(GroupCategoryEntry)
            .filter(GroupCategoryEntry.iteration_id == 1)
            .all()
        )

        # Should have 6 entries: 3 for "the wizard tower" + 3 for suffixes
        assert len(new_entries) == 6

        # Group entries by folder_id
        folder_entries = {}
        for entry in new_entries:
            if entry.folder_id not in folder_entries:
                folder_entries[entry.folder_id] = []
            folder_entries[entry.folder_id].append(entry)

        # Each folder should have 2 entries (prefix + suffix)
        assert len(folder_entries[folder1.id]) == 2
        assert len(folder_entries[folder2.id]) == 2
        assert len(folder_entries[folder3.id]) == 2

        # Verify folder1 entries
        folder1_names = {e.processed_name for e in folder_entries[folder1.id]}
        assert "the wizard tower" in folder1_names
        assert "upstairs" in folder1_names

        # Verify folder2 entries
        folder2_names = {e.processed_name for e in folder_entries[folder2.id]}
        assert "the wizard tower" in folder2_names
        assert "downstairs" in folder2_names

        # Verify folder3 entries
        folder3_names = {e.processed_name for e in folder_entries[folder3.id]}
        assert "the wizard tower" in folder3_names
        assert "outside" in folder3_names

    def test_stopword_only_prefix_not_grouped(
        self, index_session, work_session, sample_run, sample_snapshot
    ):
        """
        Test Case 2 from requirements:
        Input: ["the first", "the second", "the third"]
        Expected: No grouping (stopword-only prefix "the")
        """
        # Create test folders
        folder1 = NodeFactory(
            snapshot_id=sample_snapshot.id, name="the first", kind=NodeKind.DIR
        )
        folder2 = NodeFactory(
            snapshot_id=sample_snapshot.id, name="the second", kind=NodeKind.DIR
        )
        folder3 = NodeFactory(
            snapshot_id=sample_snapshot.id, name="the third", kind=NodeKind.DIR
        )
        index_session.commit()

        # Create iteration 0
        iteration = GroupIteration(
            id=0,
            run_id=sample_run.id,
            snapshot_id=sample_snapshot.id,
            description="Test iteration",
        )
        work_session.add(iteration)
        work_session.flush()  # Flush to satisfy foreign key constraint

        # Create initial entries
        entries = [
            GroupCategoryEntry(
                folder_id=folder1.id,
                iteration_id=0,
                pre_processed_name="the first",
                processed_name="the first",
                path=str(folder1.abs_path),
                confidence=1.0,
                processed=False,
            ),
            GroupCategoryEntry(
                folder_id=folder2.id,
                iteration_id=0,
                pre_processed_name="the second",
                processed_name="the second",
                path=str(folder2.abs_path),
                confidence=1.0,
                processed=False,
            ),
            GroupCategoryEntry(
                folder_id=folder3.id,
                iteration_id=0,
                pre_processed_name="the third",
                processed_name="the third",
                path=str(folder3.abs_path),
                confidence=1.0,
                processed=False,
            ),
        ]
        for entry in entries:
            work_session.add(entry)
        work_session.commit()

        # Apply folder name grouping
        apply_folder_name_grouping(
            work_session, run_id=sample_run.id, snapshot_id=sample_snapshot.id
        )

        # Verify results - should have same entries ungrouped
        new_entries = (
            work_session.query(GroupCategoryEntry)
            .filter(GroupCategoryEntry.iteration_id == 1)
            .all()
        )

        # Should have 3 entries (unchanged)
        assert len(new_entries) == 3

        # Verify names are unchanged
        processed_names = {e.processed_name for e in new_entries}
        assert processed_names == {"the first", "the second", "the third"}

    def test_apple_grouping(
        self, index_session, work_session, sample_run, sample_snapshot
    ):
        """
        Test common prefix grouping adapted from test_common_token_grouping.
        Input: ["apple pie", "apple tart", "apple juice"]
        Expected: Groups for "apple" and individual suffixes
        """
        # Create test folders
        folder1 = NodeFactory(
            snapshot_id=sample_snapshot.id, name="apple pie", kind=NodeKind.DIR
        )
        folder2 = NodeFactory(
            snapshot_id=sample_snapshot.id, name="apple tart", kind=NodeKind.DIR
        )
        folder3 = NodeFactory(
            snapshot_id=sample_snapshot.id, name="apple juice", kind=NodeKind.DIR
        )
        index_session.commit()

        # Create iteration 0
        iteration = GroupIteration(
            id=0,
            run_id=sample_run.id,
            snapshot_id=sample_snapshot.id,
            description="Test iteration",
        )
        work_session.add(iteration)
        work_session.flush()  # Flush to satisfy foreign key constraint

        # Create initial entries
        entries = [
            GroupCategoryEntry(
                folder_id=folder1.id,
                iteration_id=0,
                pre_processed_name="apple pie",
                processed_name="apple pie",
                path=str(folder1.abs_path),
                confidence=1.0,
                processed=False,
            ),
            GroupCategoryEntry(
                folder_id=folder2.id,
                iteration_id=0,
                pre_processed_name="apple tart",
                processed_name="apple tart",
                path=str(folder2.abs_path),
                confidence=1.0,
                processed=False,
            ),
            GroupCategoryEntry(
                folder_id=folder3.id,
                iteration_id=0,
                pre_processed_name="apple juice",
                processed_name="apple juice",
                path=str(folder3.abs_path),
                confidence=1.0,
                processed=False,
            ),
        ]
        for entry in entries:
            work_session.add(entry)
        work_session.commit()

        # Apply folder name grouping
        apply_folder_name_grouping(
            work_session, run_id=sample_run.id, snapshot_id=sample_snapshot.id
        )

        # Verify results
        new_entries = (
            work_session.query(GroupCategoryEntry)
            .filter(GroupCategoryEntry.iteration_id == 1)
            .all()
        )

        # Should have 6 entries: 3 for "apple" + 3 for suffixes
        assert len(new_entries) == 6

        # Group by folder and verify
        folder_entries = {}
        for entry in new_entries:
            if entry.folder_id not in folder_entries:
                folder_entries[entry.folder_id] = []
            folder_entries[entry.folder_id].append(entry)

        # Verify each folder has 2 entries
        for folder_id in [folder1.id, folder2.id, folder3.id]:
            assert len(folder_entries[folder_id]) == 2

        # Verify specific groupings
        folder1_names = {e.processed_name for e in folder_entries[folder1.id]}
        assert "apple" in folder1_names
        assert "pie" in folder1_names

    def test_no_common_prefix(
        self, index_session, work_session, sample_run, sample_snapshot
    ):
        """
        Test with no common prefix.
        Input: ["apple", "banana", "cherry"]
        Expected: No grouping
        """
        # Create test folders
        folder1 = NodeFactory(
            snapshot_id=sample_snapshot.id, name="apple", kind=NodeKind.DIR
        )
        folder2 = NodeFactory(
            snapshot_id=sample_snapshot.id, name="banana", kind=NodeKind.DIR
        )
        folder3 = NodeFactory(
            snapshot_id=sample_snapshot.id, name="cherry", kind=NodeKind.DIR
        )
        index_session.commit()

        # Create iteration 0
        iteration = GroupIteration(
            id=0,
            run_id=sample_run.id,
            snapshot_id=sample_snapshot.id,
            description="Test iteration",
        )
        work_session.add(iteration)
        work_session.flush()  # Flush to satisfy foreign key constraint

        # Create initial entries
        entries = [
            GroupCategoryEntry(
                folder_id=folder1.id,
                iteration_id=0,
                pre_processed_name="apple",
                processed_name="apple",
                path=str(folder1.abs_path),
                confidence=1.0,
                processed=False,
            ),
            GroupCategoryEntry(
                folder_id=folder2.id,
                iteration_id=0,
                pre_processed_name="banana",
                processed_name="banana",
                path=str(folder2.abs_path),
                confidence=1.0,
                processed=False,
            ),
            GroupCategoryEntry(
                folder_id=folder3.id,
                iteration_id=0,
                pre_processed_name="cherry",
                processed_name="cherry",
                path=str(folder3.abs_path),
                confidence=1.0,
                processed=False,
            ),
        ]
        for entry in entries:
            work_session.add(entry)
        work_session.commit()

        # Apply folder name grouping
        apply_folder_name_grouping(
            work_session, run_id=sample_run.id, snapshot_id=sample_snapshot.id
        )

        # Verify results - should remain unchanged
        new_entries = (
            work_session.query(GroupCategoryEntry)
            .filter(GroupCategoryEntry.iteration_id == 1)
            .all()
        )

        assert len(new_entries) == 3
        processed_names = {e.processed_name for e in new_entries}
        assert processed_names == {"apple", "banana", "cherry"}

    def test_partial_overlap(
        self, index_session, work_session, sample_run, sample_snapshot
    ):
        """
        Test partial overlap grouping.
        Input: ["apple pie", "apple tart", "banana pie"]
        Expected: Only "apple pie" and "apple tart" grouped
        """
        # Create test folders
        folder1 = NodeFactory(
            snapshot_id=sample_snapshot.id, name="apple pie", kind=NodeKind.DIR
        )
        folder2 = NodeFactory(
            snapshot_id=sample_snapshot.id, name="apple tart", kind=NodeKind.DIR
        )
        folder3 = NodeFactory(
            snapshot_id=sample_snapshot.id, name="banana pie", kind=NodeKind.DIR
        )
        index_session.commit()

        # Create iteration 0
        iteration = GroupIteration(
            id=0,
            run_id=sample_run.id,
            snapshot_id=sample_snapshot.id,
            description="Test iteration",
        )
        work_session.add(iteration)
        work_session.flush()  # Flush to satisfy foreign key constraint

        # Create initial entries
        entries = [
            GroupCategoryEntry(
                folder_id=folder1.id,
                iteration_id=0,
                pre_processed_name="apple pie",
                processed_name="apple pie",
                path=str(folder1.abs_path),
                confidence=1.0,
                processed=False,
            ),
            GroupCategoryEntry(
                folder_id=folder2.id,
                iteration_id=0,
                pre_processed_name="apple tart",
                processed_name="apple tart",
                path=str(folder2.abs_path),
                confidence=1.0,
                processed=False,
            ),
            GroupCategoryEntry(
                folder_id=folder3.id,
                iteration_id=0,
                pre_processed_name="banana pie",
                processed_name="banana pie",
                path=str(folder3.abs_path),
                confidence=1.0,
                processed=False,
            ),
        ]
        for entry in entries:
            work_session.add(entry)
        work_session.commit()

        # Apply folder name grouping
        apply_folder_name_grouping(
            work_session, run_id=sample_run.id, snapshot_id=sample_snapshot.id
        )

        # Verify results
        new_entries = (
            work_session.query(GroupCategoryEntry)
            .filter(GroupCategoryEntry.iteration_id == 1)
            .all()
        )

        # Should have 5 entries: 2 for "apple" + 2 for suffixes + 1 ungrouped "banana pie"
        assert len(new_entries) == 5

        # Find the banana pie entry (should be ungrouped)
        banana_entries = [e for e in new_entries if e.folder_id == folder3.id]
        assert len(banana_entries) == 1
        assert banana_entries[0].processed_name == "banana pie"

        # Verify apple entries are grouped
        apple_pie_entries = [e for e in new_entries if e.folder_id == folder1.id]
        assert len(apple_pie_entries) == 2
        apple_pie_names = {e.processed_name for e in apple_pie_entries}
        assert "apple" in apple_pie_names
        assert "pie" in apple_pie_names

    def test_mixed_case_handling(
        self, index_session, work_session, sample_run, sample_snapshot
    ):
        """
        Test that grouping is case-insensitive.
        Input: ["Apple Pie", "apple tart", "APPLE juice"]
        Expected: All grouped under "apple"
        """
        # Create test folders
        folder1 = NodeFactory(
            snapshot_id=sample_snapshot.id, name="Apple Pie", kind=NodeKind.DIR
        )
        folder2 = NodeFactory(
            snapshot_id=sample_snapshot.id, name="apple tart", kind=NodeKind.DIR
        )
        folder3 = NodeFactory(
            snapshot_id=sample_snapshot.id, name="APPLE juice", kind=NodeKind.DIR
        )
        index_session.commit()

        # Create iteration 0
        iteration = GroupIteration(
            id=0,
            run_id=sample_run.id,
            snapshot_id=sample_snapshot.id,
            description="Test iteration",
        )
        work_session.add(iteration)
        work_session.flush()  # Flush to satisfy foreign key constraint

        # Create initial entries
        entries = [
            GroupCategoryEntry(
                folder_id=folder1.id,
                iteration_id=0,
                pre_processed_name="Apple Pie",
                processed_name="Apple Pie",
                path=str(folder1.abs_path),
                confidence=1.0,
                processed=False,
            ),
            GroupCategoryEntry(
                folder_id=folder2.id,
                iteration_id=0,
                pre_processed_name="apple tart",
                processed_name="apple tart",
                path=str(folder2.abs_path),
                confidence=1.0,
                processed=False,
            ),
            GroupCategoryEntry(
                folder_id=folder3.id,
                iteration_id=0,
                pre_processed_name="APPLE juice",
                processed_name="APPLE juice",
                path=str(folder3.abs_path),
                confidence=1.0,
                processed=False,
            ),
        ]
        for entry in entries:
            work_session.add(entry)
        work_session.commit()

        # Apply folder name grouping
        apply_folder_name_grouping(
            work_session, run_id=sample_run.id, snapshot_id=sample_snapshot.id
        )

        # Verify results - should have 6 entries
        new_entries = (
            work_session.query(GroupCategoryEntry)
            .filter(GroupCategoryEntry.iteration_id == 1)
            .all()
        )

        assert len(new_entries) == 6

        # Each folder should have 2 entries
        for folder_id in [folder1.id, folder2.id, folder3.id]:
            folder_entries = [e for e in new_entries if e.folder_id == folder_id]
            assert len(folder_entries) == 2

    def test_empty_input(self, work_session, sample_run, sample_snapshot):
        """Test with no entries"""
        # Create iteration 0 with no entries
        iteration = GroupIteration(
            id=0,
            run_id=sample_run.id,
            snapshot_id=sample_snapshot.id,
            description="Test iteration",
        )
        work_session.add(iteration)
        work_session.commit()

        # Apply folder name grouping
        apply_folder_name_grouping(
            work_session, run_id=sample_run.id, snapshot_id=sample_snapshot.id
        )

        # Verify no new entries created
        new_entries = (
            work_session.query(GroupCategoryEntry)
            .filter(GroupCategoryEntry.iteration_id == 1)
            .all()
        )
        assert len(new_entries) == 0

    def test_confidence_preservation(
        self, index_session, work_session, sample_run, sample_snapshot
    ):
        """Test that confidence scores are preserved through grouping"""
        # Create test folders
        folder1 = NodeFactory(
            snapshot_id=sample_snapshot.id, name="apple pie", kind=NodeKind.DIR
        )
        folder2 = NodeFactory(
            snapshot_id=sample_snapshot.id, name="apple tart", kind=NodeKind.DIR
        )
        index_session.commit()

        # Create iteration 0
        iteration = GroupIteration(
            id=0,
            run_id=sample_run.id,
            snapshot_id=sample_snapshot.id,
            description="Test iteration",
        )
        work_session.add(iteration)
        work_session.flush()  # Flush to satisfy foreign key constraint

        # Create initial entries with different confidence scores
        entries = [
            GroupCategoryEntry(
                folder_id=folder1.id,
                iteration_id=0,
                pre_processed_name="apple pie",
                processed_name="apple pie",
                path=str(folder1.abs_path),
                confidence=0.8,
                processed=False,
            ),
            GroupCategoryEntry(
                folder_id=folder2.id,
                iteration_id=0,
                pre_processed_name="apple tart",
                processed_name="apple tart",
                path=str(folder2.abs_path),
                confidence=0.9,
                processed=False,
            ),
        ]
        for entry in entries:
            work_session.add(entry)
        work_session.commit()

        # Apply folder name grouping
        apply_folder_name_grouping(
            work_session, run_id=sample_run.id, snapshot_id=sample_snapshot.id
        )

        # Verify confidence scores are preserved
        new_entries = (
            work_session.query(GroupCategoryEntry)
            .filter(GroupCategoryEntry.iteration_id == 1)
            .all()
        )

        folder1_entries = [e for e in new_entries if e.folder_id == folder1.id]
        for entry in folder1_entries:
            assert entry.confidence == 0.8

        folder2_entries = [e for e in new_entries if e.folder_id == folder2.id]
        for entry in folder2_entries:
            assert entry.confidence == 0.9
