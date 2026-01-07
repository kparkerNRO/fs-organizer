"""Tests for dual representation building utilities."""

from pathlib import Path

import pytest

from storage.factories import (
    GroupCategoryEntryFactory,
    GroupCategoryFactory,
    GroupIterationFactory,
    NodeFactory,
    RunFactory,
    SnapshotFactory,
)
from storage.manager import NodeKind, StorageManager
from utils.dual_representation import build_dual_representation


@pytest.fixture
def storage_manager(tmp_path: Path) -> StorageManager:
    """Create a temporary storage manager for testing."""
    return StorageManager(database_path=tmp_path)


@pytest.fixture
def setup_test_data(storage_manager: StorageManager):
    """Setup test data in both index and work databases."""
    with storage_manager.get_index_session() as index_session:
        # Create snapshot
        SnapshotFactory._meta.sqlalchemy_session = index_session  # type: ignore[misc]
        snapshot = SnapshotFactory(
            id=1,
            root_path="/test",
            root_abs_path="/test",
        )

        # Create nodes
        NodeFactory._meta.sqlalchemy_session = index_session  # type: ignore[misc]
        root_node = NodeFactory(
            id=1,
            snapshot_id=snapshot.id,
            name="root",
            kind=NodeKind.DIR,
            parent_node_id=None,
            abs_path="/test",
            rel_path=".",
        )
        child_node1 = NodeFactory(
            id=2,
            snapshot_id=snapshot.id,
            name="docs",
            kind=NodeKind.DIR,
            parent_node_id=root_node.id,
            abs_path="/test/docs",
            rel_path="docs",
        )
        child_node2 = NodeFactory(
            id=3,
            snapshot_id=snapshot.id,
            name="file.txt",
            kind=NodeKind.FILE,
            parent_node_id=child_node1.id,
            abs_path="/test/docs/file.txt",
            rel_path="docs/file.txt",
        )
        index_session.commit()

    with storage_manager.get_work_session() as work_session:
        # Create run
        RunFactory._meta.sqlalchemy_session = work_session  # type: ignore[misc]
        run = RunFactory(id=1, snapshot_id=snapshot.id)

        # Create group iteration
        GroupIterationFactory._meta.sqlalchemy_session = work_session  # type: ignore[misc]
        iteration = GroupIterationFactory(
            id=1,
            run_id=run.id,
            snapshot_id=snapshot.id,
        )

        # Create categories
        GroupCategoryFactory._meta.sqlalchemy_session = work_session  # type: ignore[misc]
        category1 = GroupCategoryFactory(
            id=1,
            iteration_id=iteration.id,
            name="Work Documents",
            count=1,
        )

        # Create category entries (mapping nodes to categories)
        GroupCategoryEntryFactory._meta.sqlalchemy_session = work_session  # type: ignore[misc]
        GroupCategoryEntryFactory(
            id=1,
            folder_id=child_node1.id,
            group_id=category1.id,
            iteration_id=iteration.id,
            processed_name="Work Documents",
        )

        work_session.commit()

    return snapshot.id, run.id


class TestBuildDualRepresentation:
    """Test the build_dual_representation function."""

    def test_empty_database(self, storage_manager: StorageManager):
        """Test building dual representation with no data."""
        with storage_manager.get_index_session() as index_session:
            SnapshotFactory._meta.sqlalchemy_session = index_session  # type: ignore[misc]
            snapshot = SnapshotFactory(id=1)
            index_session.commit()
            snapshot_id = snapshot.id

        with storage_manager.get_work_session() as work_session:
            RunFactory._meta.sqlalchemy_session = work_session  # type: ignore[misc]
            run = RunFactory(id=1, snapshot_id=snapshot_id)
            work_session.commit()
            run_id = run.id

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot_id,
            run_id=run_id,
        )

        assert dual_rep is not None
        assert isinstance(dual_rep.items, dict)
        assert isinstance(dual_rep.node_hierarchy, dict)
        assert isinstance(dual_rep.category_hierarchy, dict)

        # Should have at least the root nodes
        assert "node-root" in dual_rep.items
        assert "category-root" in dual_rep.items

    def test_with_nodes_only(self, storage_manager: StorageManager):
        """Test building dual representation with only nodes (no categories)."""
        with storage_manager.get_index_session() as index_session:
            SnapshotFactory._meta.sqlalchemy_session = index_session  # type: ignore[misc]
            snapshot = SnapshotFactory(id=1)

            NodeFactory._meta.sqlalchemy_session = index_session  # type: ignore[misc]
            node = NodeFactory(
                id=1,
                snapshot_id=snapshot.id,
                name="test_node",
                kind=NodeKind.DIR,
            )
            index_session.commit()

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot.id,
            run_id=None,
        )

        # Check that node was added
        assert "node-1" in dual_rep.items
        assert dual_rep.items["node-1"].name == "test_node"
        assert dual_rep.items["node-1"].type == "node"

        # Should have hierarchy entry for root
        assert "node-root" in dual_rep.node_hierarchy
        assert "node-1" in dual_rep.node_hierarchy["node-root"]

    def test_with_full_data(self, storage_manager: StorageManager, setup_test_data):
        """Test building dual representation with complete data."""
        snapshot_id, run_id = setup_test_data

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot_id,
            run_id=run_id,
        )

        # Check nodes are present
        assert "node-1" in dual_rep.items  # root
        assert "node-2" in dual_rep.items  # docs
        assert "node-3" in dual_rep.items  # file.txt

        # Check categories are present
        assert "category-1" in dual_rep.items  # Work Documents

        # Check node hierarchy
        assert "node-root" in dual_rep.node_hierarchy
        assert "node-1" in dual_rep.node_hierarchy["node-root"]
        assert "node-2" in dual_rep.node_hierarchy["node-1"]
        assert "node-3" in dual_rep.node_hierarchy["node-2"]

        # Check category hierarchy
        assert "category-root" in dual_rep.category_hierarchy
        assert "category-1" in dual_rep.category_hierarchy["category-root"]

    def test_node_properties(self, storage_manager: StorageManager, setup_test_data):
        """Test that node items have correct properties."""
        snapshot_id, run_id = setup_test_data

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot_id,
            run_id=run_id,
        )

        # Check file node
        file_item = dual_rep.items["node-3"]
        assert file_item.type == "node"
        assert file_item.name == "file.txt"
        assert file_item.originalPath == "/test/docs/file.txt"

        # Check directory node
        dir_item = dual_rep.items["node-2"]
        assert dir_item.type == "node"
        assert dir_item.name == "docs"
        assert dir_item.originalPath == "/test/docs"

    def test_category_properties(self, storage_manager: StorageManager, setup_test_data):
        """Test that category items have correct properties."""
        snapshot_id, run_id = setup_test_data

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot_id,
            run_id=run_id,
        )

        # Check category
        category_item = dual_rep.items["category-1"]
        assert category_item.type == "category"
        assert category_item.name == "Work Documents"
        assert category_item.originalPath is None

    def test_hierarchy_structure(self, storage_manager: StorageManager, setup_test_data):
        """Test that hierarchies are correctly structured."""
        snapshot_id, run_id = setup_test_data

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot_id,
            run_id=run_id,
        )

        # Node hierarchy should be tree-like
        assert isinstance(dual_rep.node_hierarchy["node-root"], list)
        assert isinstance(dual_rep.node_hierarchy["node-1"], list)

        # Category hierarchy should have categories under root
        assert isinstance(dual_rep.category_hierarchy["category-root"], list)
        assert "category-1" in dual_rep.category_hierarchy["category-root"]

    def test_without_run_id(self, storage_manager: StorageManager):
        """Test building without run_id (no categories)."""
        with storage_manager.get_index_session() as index_session:
            SnapshotFactory._meta.sqlalchemy_session = index_session  # type: ignore[misc]
            snapshot = SnapshotFactory(id=1)

            NodeFactory._meta.sqlalchemy_session = index_session  # type: ignore[misc]
            NodeFactory(id=1, snapshot_id=snapshot.id, name="test", kind=NodeKind.DIR)
            index_session.commit()

        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=snapshot.id,
            run_id=None,
        )

        # Should have nodes but category hierarchy should only have root
        assert "node-1" in dual_rep.items
        assert "category-root" in dual_rep.items
        assert len(dual_rep.category_hierarchy["category-root"]) == 0
