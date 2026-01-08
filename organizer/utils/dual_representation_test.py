"""Tests for dual representation building utilities."""

import pytest

from storage.factories import (
    GroupCategoryEntryFactory,
    GroupCategoryFactory,
    GroupIterationFactory,
    NodeFactory,
    RunFactory,
    SnapshotFactory,
)
from storage.manager import NodeKind
from utils.dual_representation import build_dual_representation


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

    def test_empty_database(
        self, storage_manager, storage_index_session, storage_work_session
    ):
        """Test building dual representation with no data."""
        snapshot = SnapshotFactory()
        storage_index_session.commit()

        run = RunFactory(snapshot_id=snapshot.id)
        storage_work_session.commit()

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot.id,
            run_id=run.id,
        )

        assert dual_rep is not None
        assert isinstance(dual_rep.items, dict)
        assert isinstance(dual_rep.node_hierarchy, dict)
        assert isinstance(dual_rep.category_hierarchy, dict)

        # Should have at least the root nodes
        assert "node-root" in dual_rep.items
        assert "category-root" in dual_rep.items

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
        )

        # Check that node was added
        node_key = f"node-{node.id}"
        assert node_key in dual_rep.items
        assert dual_rep.items[node_key].name == "test_node"
        assert dual_rep.items[node_key].type == "node"

        # Should have hierarchy entry for root
        assert "node-root" in dual_rep.node_hierarchy
        assert node_key in dual_rep.node_hierarchy["node-root"]

    def test_with_full_data(self, storage_manager, setup_test_data):
        """Test building dual representation with complete data."""
        snapshot_id, run_id = setup_test_data

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot_id,
            run_id=run_id,
        )

        # Check nodes are present (verify count rather than specific IDs)
        node_items = [item for item in dual_rep.items.values() if item.type == "node"]
        assert len(node_items) == 4  # root + docs + file.txt + node-root

        # Check categories are present
        category_items = [
            item for item in dual_rep.items.values() if item.type == "category"
        ]
        assert len(category_items) == 2  # Work Documents + category-root

        # Check hierarchies exist
        assert "node-root" in dual_rep.node_hierarchy
        assert "category-root" in dual_rep.category_hierarchy

    def test_node_properties(self, storage_manager, setup_test_data):
        """Test that node items have correct properties."""
        snapshot_id, run_id = setup_test_data

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot_id,
            run_id=run_id,
        )

        # Find file node by name
        file_item = next(
            item
            for item in dual_rep.items.values()
            if item.type == "node" and item.name == "file.txt"
        )
        assert file_item.type == "node"
        assert file_item.name == "file.txt"
        assert file_item.originalPath == "/test/docs/file.txt"

        # Find directory node by name
        dir_item = next(
            item
            for item in dual_rep.items.values()
            if item.type == "node" and item.name == "docs"
        )
        assert dir_item.type == "node"
        assert dir_item.name == "docs"
        assert dir_item.originalPath == "/test/docs"

    def test_category_properties(self, storage_manager, setup_test_data):
        """Test that category items have correct properties."""
        snapshot_id, run_id = setup_test_data

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot_id,
            run_id=run_id,
        )

        # Find category by name
        category_item = next(
            item
            for item in dual_rep.items.values()
            if item.type == "category" and item.name == "Work Documents"
        )
        assert category_item.type == "category"
        assert category_item.name == "Work Documents"
        assert category_item.originalPath is None

    def test_hierarchy_structure(self, storage_manager, setup_test_data):
        """Test that hierarchies are correctly structured."""
        snapshot_id, run_id = setup_test_data

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot_id,
            run_id=run_id,
        )

        # Node hierarchy should be tree-like
        assert isinstance(dual_rep.node_hierarchy["node-root"], list)
        assert len(dual_rep.node_hierarchy["node-root"]) > 0

        # Category hierarchy should have categories under root
        assert isinstance(dual_rep.category_hierarchy["category-root"], list)
        assert len(dual_rep.category_hierarchy["category-root"]) == 1

    def test_without_run_id(
        self, storage_manager, storage_index_session, storage_work_session
    ):
        """Test building without run_id (no categories)."""
        snapshot = SnapshotFactory()
        node = NodeFactory(snapshot_id=snapshot.id, name="test", kind=NodeKind.DIR)
        storage_index_session.commit()

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot.id,
            run_id=None,
        )

        # Should have nodes but category hierarchy should only have root
        node_key = f"node-{node.id}"
        assert node_key in dual_rep.items
        assert "category-root" in dual_rep.items
        assert len(dual_rep.category_hierarchy["category-root"]) == 0
