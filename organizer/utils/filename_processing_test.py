from api.api import PipelineStage
from sqlalchemy import select
from storage.factories import GroupCategoryEntryFactory, NodeFactory
from storage.manager import NodeKind
from storage.work_models import FileMapping

from organizer.utils.filename_processing import (
    calculate_cleaned_paths_from_groups,
)


def test_recalculate_cleaned_paths_creates_and_updates_mappings(
    storage_manager,
    sample_snapshot,
    sample_run,
    index_session,
    work_session,
):
    folder = NodeFactory(
        snapshot_id=sample_snapshot.id,
        kind=NodeKind.DIR.value,
        name="My_Folder",
        rel_path="My_Folder",
        abs_path="/test/My_Folder",
        depth=1,
    )
    archive = NodeFactory(
        snapshot_id=sample_snapshot.id,
        kind=NodeKind.DIR.value,
        name="Archive.zip",
        rel_path="Archive.zip",
        abs_path="/test/Archive.zip",
        depth=1,
    )

    work_session.add(
        FileMapping(
            run_id=sample_run.id,
            node_id=archive.id,
            original_path=archive.abs_path,
            new_path="old-path",
        )
    )
    work_session.commit()

    updated = calculate_cleaned_paths_from_groups(
        storage_manager,
        sample_snapshot.id,
        sample_run.id,
        structure_type=PipelineStage.original,
    )

    assert updated == 2

    mappings = {
        mapping.node_id: mapping.new_path
        for mapping in work_session.execute(
            select(FileMapping).where(FileMapping.run_id == sample_run.id)
        )
        .scalars()
        .all()
    }
    assert mappings[folder.id] == "My Folder"
    assert mappings[archive.id] == "Archive"


def test_recalculate_cleaned_paths_for_structure_organized_uses_categories(
    storage_manager,
    sample_snapshot,
    sample_run,
    sample_iteration,
    index_session,
    work_session,
):
    parent = NodeFactory(
        snapshot_id=sample_snapshot.id,
        kind=NodeKind.DIR.value,
        name="parent",
        rel_path="parent",
        abs_path="/test/parent",
        depth=1,
    )
    child = NodeFactory(
        snapshot_id=sample_snapshot.id,
        kind=NodeKind.DIR.value,
        name="child",
        rel_path="parent/child",
        abs_path="/test/parent/child",
        depth=2,
        parent_node_id=parent.id,
    )

    GroupCategoryEntryFactory(
        folder_id=parent.id,
        iteration=sample_iteration,
        processed_name="art_category",
        confidence=0.9,
    )

    work_session.add(
        FileMapping(
            run_id=sample_run.id,
            node_id=child.id,
            original_path=child.abs_path,
            new_path="old-path",
        )
    )
    work_session.commit()

    updated = calculate_cleaned_paths_from_groups(
        storage_manager,
        sample_snapshot.id,
        sample_run.id,
        PipelineStage.organized,
    )

    assert updated == 2

    mappings = {
        mapping.node_id: mapping.new_path
        for mapping in work_session.execute(
            select(FileMapping).where(FileMapping.run_id == sample_run.id)
        )
        .scalars()
        .all()
    }
    assert mappings[parent.id] == ""
    assert mappings[child.id] == "art_category"
