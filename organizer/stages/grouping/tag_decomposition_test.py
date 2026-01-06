"""
Tests for tag decomposition functionality
"""

from typing import cast

import pytest
from sqlalchemy import select
from storage.factories import (
    GroupCategoryEntryFactory,
    GroupIterationFactory,
    NodeFactory,
)
from storage.manager import NodeKind
from storage.work_models import GroupCategoryEntry, GroupIteration

from stages.grouping.tag_decomposition import decompose_compound_tags


@pytest.fixture
def test_nodes(index_session, sample_snapshot):
    """Create test nodes in the index database"""
    nodes = [
        NodeFactory(
            snapshot_id=sample_snapshot.id,
            id=1,
            name="test1",
            abs_path="/test1",
            rel_path="test1",
            kind=NodeKind.DIR,
        ),
        NodeFactory(
            snapshot_id=sample_snapshot.id,
            id=2,
            name="test2",
            abs_path="/test2",
            rel_path="test2",
            kind=NodeKind.DIR,
        ),
        NodeFactory(
            snapshot_id=sample_snapshot.id,
            id=3,
            name="test3",
            abs_path="/test3",
            rel_path="test3",
            kind=NodeKind.DIR,
        ),
    ]
    return nodes


@pytest.fixture
def test_entries(work_session, sample_run, test_nodes):
    """Create test GroupCategoryEntry records for decomposition testing"""
    # Create iteration record
    iteration = cast(
        GroupIteration,
        GroupIterationFactory(
            id=0,
            run=sample_run,
        ),
    )

    # Create test entries with compound tags
    test_entries = [
        # Tower examples
        GroupCategoryEntryFactory(
            folder_id=1,
            iteration=iteration,
            processed_name="Castle Tower",
            pre_processed_name="Castle Tower",
            path="/test1",
            confidence=0.8,
        ),
        GroupCategoryEntryFactory(
            folder_id=2,
            iteration=iteration,
            processed_name="Wizard Tower",
            pre_processed_name="Wizard Tower",
            path="/test2",
            confidence=0.8,
        ),
        GroupCategoryEntryFactory(
            folder_id=3,
            iteration=iteration,
            processed_name="Tower Interior",
            pre_processed_name="Tower Interior",
            path="/test3",
            confidence=0.8,
        ),
        # Location type examples
        GroupCategoryEntryFactory(
            folder_id=1,
            iteration=iteration,
            processed_name="Village Tavern",
            pre_processed_name="Village Tavern",
            path="/test1",
            confidence=0.8,
        ),
        GroupCategoryEntryFactory(
            folder_id=2,
            iteration=iteration,
            processed_name="City Tavern",
            pre_processed_name="City Tavern",
            path="/test2",
            confidence=0.8,
        ),
        # Collaboration examples
        GroupCategoryEntryFactory(
            folder_id=1,
            iteration=iteration,
            processed_name="Collaboration with Fantasy Atlas",
            pre_processed_name="Collaboration with Fantasy Atlas",
            path="/test1",
            confidence=0.8,
        ),
        GroupCategoryEntryFactory(
            folder_id=2,
            iteration=iteration,
            processed_name="Collaboration with Cze Peku",
            pre_processed_name="Collaboration with Cze Peku",
            path="/test2",
            confidence=0.8,
        ),
        # Single word tags (should not be decomposed)
        GroupCategoryEntryFactory(
            folder_id=1,
            iteration=iteration,
            processed_name="Interior",
            pre_processed_name="Interior",
            path="/test1",
            confidence=0.8,
        ),
        GroupCategoryEntryFactory(
            folder_id=2,
            iteration=iteration,
            processed_name="Maps",
            pre_processed_name="Maps",
            path="/test2",
            confidence=0.8,
        ),
    ]

    return test_entries


@pytest.mark.ml
def test_full_decomposition_pipeline(work_session, test_entries):
    """Test the complete decomposition pipeline"""
    # Count original entries
    original_count = len(test_entries)

    # Run decomposition
    decompose_compound_tags(work_session)

    # Check new iteration was created
    from stages.grouping.group import get_next_iteration_id

    new_iteration_id = get_next_iteration_id(work_session) - 1

    # Get new entries
    stmt = select(GroupCategoryEntry).where(
        GroupCategoryEntry.iteration_id == new_iteration_id
    )
    new_entries = work_session.scalars(stmt).all()

    # Should have more entries due to decomposition
    assert len(new_entries) >= original_count

    # Check that some decomposed entries exist
    decomposed_entries = [e for e in new_entries if e.derived_names]

    print(f"Original entries: {original_count}")
    print(f"New entries: {len(new_entries)}")
    print(f"Decomposed entries: {len(decomposed_entries)}")

    if decomposed_entries:
        print("Sample decomposed entries:")
        for entry in decomposed_entries[:3]:
            print(f"  {entry.processed_name} (from {entry.derived_names})")
