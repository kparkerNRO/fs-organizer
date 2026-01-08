"""Tests for dual representation building utilities."""

import pytest

from data_models.pipeline import PipelineStage
from storage.factories import (
    FileMappingFactory,
    GroupCategoryEntryFactory,
    GroupCategoryFactory,
    GroupIterationFactory,
    NodeFactory,
    RunFactory,
    SnapshotFactory,
)
from storage.manager import NodeKind
from utils.dual_representation import (
    build_dual_representation,
    dual_representation_to_folder_structure,
)


@pytest.fixture
def setup_test_data(storage_index_session, storage_work_session):
    """Setup test data in both index and work databases."""
    # Create snapshot
    snapshot = SnapshotFactory()

    # Create nodes
    root_node = NodeFactory(
        snapshot_id=snapshot.id,
        name="root",
        kind=NodeKind.DIR,
        parent_node_id=None,
        abs_path="/test",
        rel_path=".",
    )
    child_node1 = NodeFactory(
        snapshot_id=snapshot.id,
        name="docs",
        kind=NodeKind.DIR,
        parent_node_id=root_node.id,
        abs_path="/test/docs",
        rel_path="docs",
    )
    NodeFactory(
        snapshot_id=snapshot.id,
        name="file.txt",
        kind=NodeKind.FILE,
        parent_node_id=child_node1.id,
        abs_path="/test/docs/file.txt",
        rel_path="docs/file.txt",
    )
    storage_index_session.commit()

    # Create run with factories
    run = RunFactory(snapshot_id=snapshot.id)

    # Create group iteration
    iteration = GroupIterationFactory(run=run, snapshot_id=snapshot.id)

    # Create category
    category = GroupCategoryFactory(iteration=iteration, name="Work Documents")

    # Create category entry
    GroupCategoryEntryFactory(
        folder_id=child_node1.id,
        group_id=category.id,
        iteration=iteration,
        processed_name="Work Documents",
    )

    storage_work_session.commit()

    return snapshot.id, run.id


class TestBuildDualRepresentation:
    """Test the build_dual_representation function."""

    def test_empty_database_original_only(
        self, storage_manager, storage_index_session, storage_work_session
    ):
        """Test building dual representation with no data, original stage only."""
        snapshot = SnapshotFactory()
        storage_index_session.commit()

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot.id,
            run_id=None,
            stages=[PipelineStage.original],
        )

        assert dual_rep is not None
        assert isinstance(dual_rep.items, dict)
        assert isinstance(dual_rep.hierarchies, dict)

        # Should have original hierarchy
        assert "original" in dual_rep.hierarchies
        assert dual_rep.hierarchies["original"].stage == "original"
        assert dual_rep.hierarchies["original"].source_type == "node"
        assert "original-root" in dual_rep.items

    def test_with_nodes_only(
        self, storage_manager, storage_index_session, storage_work_session
    ):
        """Test building dual representation with only nodes (no categories)."""
        snapshot = SnapshotFactory()
        node = NodeFactory(
            snapshot_id=snapshot.id,
            name="test_node",
            kind=NodeKind.DIR,
        )
        storage_index_session.commit()

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot.id,
            run_id=None,
            stages=[PipelineStage.original],
        )

        # Check that node was added
        node_key = f"node-{node.id}"
        assert node_key in dual_rep.items
        assert dual_rep.items[node_key].name == "test_node"
        assert dual_rep.items[node_key].type == "node"

        # Should have hierarchy entry for root
        assert "original" in dual_rep.hierarchies
        orig_hierarchy = dual_rep.hierarchies["original"]
        assert orig_hierarchy.root.id == "original-root"
        # Check that node is in root's children
        child_ids = [child.id for child in orig_hierarchy.root.children]
        assert node_key in child_ids

    def test_with_full_data_both_stages(self, storage_manager, setup_test_data):
        """Test building dual representation with complete data for both stages."""
        snapshot_id, run_id = setup_test_data

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot_id,
            run_id=run_id,
            stages=[PipelineStage.original, PipelineStage.organized],
        )

        # Should have both hierarchies
        assert "original" in dual_rep.hierarchies
        assert "organized" in dual_rep.hierarchies

        # Check original hierarchy
        assert dual_rep.hierarchies["original"].source_type == "node"
        assert dual_rep.hierarchies["original"].root.id == "original-root"

        # Check organized hierarchy
        assert dual_rep.hierarchies["organized"].source_type == "category"
        assert dual_rep.hierarchies["organized"].root.id == "organized-root"

        # Should have items from both stages
        assert "original-root" in dual_rep.items
        assert "organized-root" in dual_rep.items

    def test_confidence_and_count_fields(self, storage_manager, setup_test_data):
        """Test that confidence and count fields are populated correctly."""
        snapshot_id, run_id = setup_test_data

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot_id,
            run_id=run_id,
            stages=[PipelineStage.original, PipelineStage.organized],
        )

        # Check that count is populated for nodes with children
        root_item = dual_rep.items["original-root"]
        assert root_item.count > 0

        # Check that confidence is set for categories
        category_item = next(
            item
            for item in dual_rep.items.values()
            if item.type == "category" and item.name == "Work Documents"
        )
        assert category_item.confidence >= 0.0
        assert category_item.confidence <= 1.0

    def test_new_path_from_file_mapping(
        self, storage_manager, storage_index_session, storage_work_session
    ):
        """Test that newPath is populated from FileMapping."""
        snapshot = SnapshotFactory()
        file_node = NodeFactory(
            snapshot_id=snapshot.id,
            name="document.pdf",
            kind=NodeKind.FILE,
        )
        storage_index_session.commit()

        run = RunFactory(snapshot_id=snapshot.id)
        iteration = GroupIterationFactory(run=run, snapshot_id=snapshot.id)
        category = GroupCategoryFactory(iteration=iteration, name="Documents")

        # Create file mapping
        FileMappingFactory(
            run_id=run.id,
            node_id=file_node.id,
            original_path="/test/document.pdf",
            new_path="Documents/document.pdf",
        )

        # Create category entry
        GroupCategoryEntryFactory(
            folder_id=file_node.id,
            group_id=category.id,
            iteration=iteration,
            processed_name="Documents",
        )

        storage_work_session.commit()

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot.id,
            run_id=run.id,
            stages=[PipelineStage.organized],
        )

        # Find the file node by ID (name in organized stage comes from processed_name)
        node_id = f"node-{file_node.id}"
        assert node_id in dual_rep.items
        file_item = dual_rep.items[node_id]
        assert file_item.newPath == "Documents/document.pdf"

    def test_compatibility_conversion_to_folder_structure(
        self, storage_manager, setup_test_data
    ):
        """Test conversion to FolderV2-compatible format."""
        snapshot_id, run_id = setup_test_data

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot_id,
            run_id=run_id,
            stages=[PipelineStage.original, PipelineStage.organized],
        )

        # Convert original stage to FolderV2 format
        folder_structure = dual_representation_to_folder_structure(
            dual_rep, stage_name="original"
        )

        # Verify structure has expected fields
        assert "id" in folder_structure
        assert "name" in folder_structure
        assert "type" in folder_structure
        assert "children" in folder_structure
        assert "count" in folder_structure
        assert "confidence" in folder_structure

        # Verify it's a valid tree
        assert isinstance(folder_structure["children"], list)
        assert len(folder_structure["children"]) > 0

    def test_compatibility_conversion_organized_stage(
        self, storage_manager, setup_test_data
    ):
        """Test conversion of organized stage to FolderV2 format."""
        snapshot_id, run_id = setup_test_data

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot_id,
            run_id=run_id,
            stages=[PipelineStage.organized],
        )

        # Convert organized hierarchy
        organized_structure = dual_representation_to_folder_structure(
            dual_rep, stage_name="organized"
        )

        # Verify structure
        assert organized_structure["name"] == "Organized"
        assert "children" in organized_structure
        assert len(organized_structure["children"]) == 1  # Work Documents category

    def test_default_stages(self, storage_manager, setup_test_data):
        """Test that default stages are original and organized."""
        snapshot_id, run_id = setup_test_data

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot_id,
            run_id=run_id,
            # No stages parameter - should default to [original, organized]
        )

        # Should have both default stages
        assert "original" in dual_rep.hierarchies
        assert "organized" in dual_rep.hierarchies
        assert len(dual_rep.hierarchies) == 2
