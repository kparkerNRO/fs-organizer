"""
Tests for tag decomposition functionality
"""

import tempfile
from pathlib import Path
from sqlalchemy.orm import Session

from data_models.database import (
    get_sessionmaker,
    setup_gather,
    setup_group,
    setup_folder_categories,
    GroupCategoryEntry,
    Folder,
)
from grouping.tag_decomposition import decompose_compound_tags


def create_test_database():
    """Create a temporary database for testing"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_path = Path(temp_file.name)
    temp_file.close()

    # Setup database tables
    setup_gather(db_path)
    setup_folder_categories(db_path)
    setup_group(db_path)

    return db_path


def create_test_entries(session: Session):
    """Create test GroupCategoryEntry records for decomposition testing"""
    # Create some test folders
    folders = [
        Folder(id=1, folder_name="test1", folder_path="/test1"),
        Folder(id=2, folder_name="test2", folder_path="/test2"),
        Folder(id=3, folder_name="test3", folder_path="/test3"),
    ]
    for folder in folders:
        session.add(folder)

    # Create test entries with compound tags
    test_entries = [
        # Tower examples
        GroupCategoryEntry(
            folder_id=1,
            iteration_id=0,
            processed_name="Castle Tower",
            pre_processed_name="Castle Tower",
            path="/test1",
            confidence=0.8,
        ),
        GroupCategoryEntry(
            folder_id=2,
            iteration_id=0,
            processed_name="Wizard Tower",
            pre_processed_name="Wizard Tower",
            path="/test2",
            confidence=0.8,
        ),
        GroupCategoryEntry(
            folder_id=3,
            iteration_id=0,
            processed_name="Tower Interior",
            pre_processed_name="Tower Interior",
            path="/test3",
            confidence=0.8,
        ),
        # Location type examples
        GroupCategoryEntry(
            folder_id=1,
            iteration_id=0,
            processed_name="Village Tavern",
            pre_processed_name="Village Tavern",
            path="/test1",
            confidence=0.8,
        ),
        GroupCategoryEntry(
            folder_id=2,
            iteration_id=0,
            processed_name="City Tavern",
            pre_processed_name="City Tavern",
            path="/test2",
            confidence=0.8,
        ),
        # Collaboration examples
        GroupCategoryEntry(
            folder_id=1,
            iteration_id=0,
            processed_name="Collaboration with Fantasy Atlas",
            pre_processed_name="Collaboration with Fantasy Atlas",
            path="/test1",
            confidence=0.8,
        ),
        GroupCategoryEntry(
            folder_id=2,
            iteration_id=0,
            processed_name="Collaboration with Cze Peku",
            pre_processed_name="Collaboration with Cze Peku",
            path="/test2",
            confidence=0.8,
        ),
        # Single word tags (should not be decomposed)
        GroupCategoryEntry(
            folder_id=1,
            iteration_id=0,
            processed_name="Interior",
            pre_processed_name="Interior",
            path="/test1",
            confidence=0.8,
        ),
        GroupCategoryEntry(
            folder_id=2,
            iteration_id=0,
            processed_name="Maps",
            pre_processed_name="Maps",
            path="/test2",
            confidence=0.8,
        ),
    ]

    for entry in test_entries:
        session.add(entry)

    session.commit()
    return test_entries


def test_full_decomposition_pipeline():
    """Test the complete decomposition pipeline"""
    db_path = create_test_database()
    sessionmaker = get_sessionmaker(db_path)

    with sessionmaker() as session:
        entries = create_test_entries(session)

        # Count original entries
        original_count = len(entries)

        # Run decomposition
        decompose_compound_tags(session)

        # Check new iteration was created
        from grouping.group import get_next_iteration_id

        new_iteration_id = get_next_iteration_id(session) - 1

        # Get new entries
        from sqlalchemy import select

        stmt = select(GroupCategoryEntry).where(
            GroupCategoryEntry.iteration_id == new_iteration_id
        )
        new_entries = session.scalars(stmt).all()

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


if __name__ == "__main__":
    # Run tests manually
    print("Testing full decomposition pipeline...")
    test_full_decomposition_pipeline()

#     print("\nAll tests completed!")
