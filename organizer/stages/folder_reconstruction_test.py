from sqlalchemy import select

from api.api import StructureType
from stages.folder_reconstruction import (
    generate_folder_heirarchy_from_path,
    recalculate_cleaned_paths,
    recalculate_cleaned_paths_for_structure,
)
from storage.factories import GroupCategoryEntryFactory, NodeFactory
from storage.manager import NodeKind
from storage.work_models import FileMapping


def test_generate_folder_heirarchy_from_path_accumulates_counts():
    working = generate_folder_heirarchy_from_path("root/sub", {})
    assert working == {"root": {"sub": {"__count__": 1}}}

    working = generate_folder_heirarchy_from_path("root/sub", working)
    working = generate_folder_heirarchy_from_path("root/other", working)

    assert working["root"]["sub"]["__count__"] == 2
    assert working["root"]["other"]["__count__"] == 1


def test_recalculate_cleaned_paths_creates_and_updates_mappings(
    storage_manager,
    sample_snapshot,
    sample_run,
    index_session,
    work_session,
):
    folder = NodeFactory(
        snapshot_id=sample_snapshot.snapshot_id,
        kind=NodeKind.DIR.value,
        name="My_Folder",
        rel_path="My_Folder",
        abs_path="/test/My_Folder",
        depth=1,
    )
    archive = NodeFactory(
        snapshot_id=sample_snapshot.snapshot_id,
        kind=NodeKind.DIR.value,
        name="Archive.zip",
        rel_path="Archive.zip",
        abs_path="/test/Archive.zip",
        depth=1,
    )

    work_session.add(
        FileMapping(
            run_id=sample_run.id,
            node_id=archive.node_id,
            original_path=archive.abs_path,
            new_path="old-path",
        )
    )
    work_session.commit()

    updated = recalculate_cleaned_paths(
        storage_manager, sample_snapshot.snapshot_id, sample_run.id
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
    assert mappings[folder.node_id] == "My Folder"
    assert mappings[archive.node_id] == "Archive"


def test_recalculate_cleaned_paths_for_structure_organized_uses_categories(
    storage_manager,
    sample_snapshot,
    sample_run,
    sample_iteration,
    index_session,
    work_session,
):
    parent = NodeFactory(
        snapshot_id=sample_snapshot.snapshot_id,
        kind=NodeKind.DIR.value,
        name="parent",
        rel_path="parent",
        abs_path="/test/parent",
        depth=1,
    )
    child = NodeFactory(
        snapshot_id=sample_snapshot.snapshot_id,
        kind=NodeKind.DIR.value,
        name="child",
        rel_path="parent/child",
        abs_path="/test/parent/child",
        depth=2,
        parent_node_id=parent.node_id,
    )

    GroupCategoryEntryFactory(
        folder_id=parent.node_id,
        iteration=sample_iteration,
        processed_name="art_category",
        confidence=0.9,
    )

    work_session.add(
        FileMapping(
            run_id=sample_run.id,
            node_id=child.node_id,
            original_path=child.abs_path,
            new_path="old-path",
        )
    )
    work_session.commit()

    updated = recalculate_cleaned_paths_for_structure(
        storage_manager,
        sample_snapshot.snapshot_id,
        sample_run.id,
        StructureType.organized,
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
    assert mappings[parent.node_id] == ""
    assert mappings[child.node_id] == "art_category"
