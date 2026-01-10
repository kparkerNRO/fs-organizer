"""Tests for dual representation building utilities.

These tests validate the implementation based on the design specified
in notes/dual_representation_design.md and test cases in
notes/dual_representation_test_cases.md.
"""

from data_models.pipeline import FolderV2, ItemType, PipelineStage
from storage.factories import (
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
    convert_hierarchy_to_folder_structure,
    create_hierarchy_from_categories,
    create_hierarchy_from_nodes,
)


class TestCreateHierarchyFromNodes:
    """Test create_hierarchy_from_nodes function (TC-BUILD-01, TC-BUILD-02, TC-BUILD-03)."""

    def test_empty_snapshot(self, storage_manager, storage_index_session):
        """TC-BUILD-01: Empty snapshot."""
        snapshot = SnapshotFactory()
        storage_index_session.commit()

        hierarchy, items = create_hierarchy_from_nodes(snapshot.id, storage_manager)  # type: ignore[attr-defined]

        # Should have root item only
        assert len(items) == 1
        root_item_id = f"snapshot-{snapshot.id}-root"
        assert root_item_id in items
        assert items[root_item_id].type == ItemType.NODE

        # Tree should have no children
        assert hierarchy.root.name == "root"
        assert len(hierarchy.root.children) == 0
        assert hierarchy.stage == PipelineStage.original
        assert hierarchy.source_type == ItemType.NODE

    def test_single_root_node(self, storage_manager, storage_index_session):
        """TC-BUILD-02: Single root node."""
        snapshot = SnapshotFactory()
        file_node = NodeFactory(
            snapshot_id=snapshot.id,
            name="file.txt",
            kind=NodeKind.FILE,
            parent_node_id=None,
            abs_path="/test/file.txt",
        )
        storage_index_session.commit()

        hierarchy, items = create_hierarchy_from_nodes(snapshot.id, storage_manager)  # type: ignore[attr-defined]

        # Should have root item and one node item
        assert len(items) == 2
        assert f"snapshot-{snapshot.id}-root" in items
        assert f"node-{file_node.id}" in items

        # Tree should have one child
        assert len(hierarchy.root.children) == 1
        child = hierarchy.root.children[0]
        assert child.itemId == f"node-{file_node.id}"
        assert child.name == "file.txt"

    def test_deeply_nested_hierarchy(self, storage_manager, storage_index_session):
        """TC-BUILD-03: Deeply nested node hierarchy."""
        snapshot = SnapshotFactory()

        # Create dir1/dir2/file.txt structure
        dir1 = NodeFactory(
            snapshot_id=snapshot.id,
            name="dir1",
            kind=NodeKind.DIR,
            parent_node_id=None,
            abs_path="/test/dir1",
        )
        dir2 = NodeFactory(
            snapshot_id=snapshot.id,
            name="dir2",
            kind=NodeKind.DIR,
            parent_node_id=dir1.id,
            abs_path="/test/dir1/dir2",
        )
        file = NodeFactory(
            snapshot_id=snapshot.id,
            name="file.txt",
            kind=NodeKind.FILE,
            parent_node_id=dir2.id,
            abs_path="/test/dir1/dir2/file.txt",
        )
        storage_index_session.commit()

        hierarchy, items = create_hierarchy_from_nodes(snapshot.id, storage_manager)  # type: ignore[attr-defined]

        # Verify items store contains all nodes with intrinsic data only
        assert len(items) == 4  # root + dir1 + dir2 + file
        assert f"node-{dir1.id}" in items
        assert f"node-{dir2.id}" in items
        assert f"node-{file.id}" in items

        # Verify items have correct type and originalPath (no name!)
        assert items[f"node-{dir1.id}"].type == ItemType.NODE
        assert items[f"node-{dir1.id}"].originalPath == "/test/dir1"
        assert not hasattr(
            items[f"node-{dir1.id}"], "name"
        )  # name is in Record, not Item

        # Verify tree structure with contextual names
        assert len(hierarchy.root.children) == 1
        dir1_record = hierarchy.root.children[0]
        assert dir1_record.itemId == f"node-{dir1.id}"
        assert dir1_record.name == "dir1"  # Name is contextual

        assert len(dir1_record.children) == 1
        dir2_record = dir1_record.children[0]
        assert dir2_record.itemId == f"node-{dir2.id}"
        assert dir2_record.name == "dir2"

        assert len(dir2_record.children) == 1
        file_record = dir2_record.children[0]
        assert file_record.itemId == f"node-{file.id}"
        assert file_record.name == "file.txt"

    def test_siblings(self, storage_manager, storage_index_session):
        """TC-BUILD-03 (continued): Test siblings in hierarchy."""
        snapshot = SnapshotFactory()

        dir1 = NodeFactory(
            snapshot_id=snapshot.id,
            name="dir1",
            kind=NodeKind.DIR,
            parent_node_id=None,
        )
        _ = NodeFactory(
            snapshot_id=snapshot.id,
            name="dir2",
            kind=NodeKind.DIR,
            parent_node_id=dir1.id,
        )
        _ = NodeFactory(
            snapshot_id=snapshot.id,
            name="dir3",
            kind=NodeKind.DIR,
            parent_node_id=dir1.id,
        )
        storage_index_session.commit()

        hierarchy, items = create_hierarchy_from_nodes(snapshot.id, storage_manager)  # type: ignore[attr-defined]

        # dir1 should have two children
        dir1_record = hierarchy.root.children[0]
        assert len(dir1_record.children) == 2

        child_names = {child.name for child in dir1_record.children}
        assert child_names == {"dir2", "dir3"}


class TestCreateHierarchyFromCategories:
    """Test create_hierarchy_from_categories function (TC-BUILD-04, TC-BUILD-05, TC-BUILD-06)."""

    def test_no_categories_exist(self, storage_manager, storage_work_session):
        """TC-BUILD-04: No categories exist."""
        run = RunFactory()
        storage_work_session.commit()

        hierarchy, items = create_hierarchy_from_categories(
            run.id,  # type: ignore[attr-defined]
            run.snapshot_id,  # type: ignore[attr-defined]
            PipelineStage.organized,
            storage_manager,
        )

        # Should have root item only
        assert len(items) == 1
        root_item_id = f"run-{run.id}-root"  # type: ignore[attr-defined]
        assert root_item_id in items

        # Tree should have no children
        assert hierarchy.root.name == "Organized"
        assert len(hierarchy.root.children) == 0

    def test_simple_category_hierarchy(
        self, storage_manager, storage_index_session, storage_work_session
    ):
        """TC-BUILD-05: Simple category hierarchy."""
        # Create nodes in index
        snapshot = SnapshotFactory()
        node1 = NodeFactory(snapshot_id=snapshot.id, name="file1.txt")
        node2 = NodeFactory(snapshot_id=snapshot.id, name="file2.txt")
        storage_index_session.commit()

        # Create run and category
        run = RunFactory(snapshot_id=snapshot.id)
        iteration = GroupIterationFactory(run=run, snapshot_id=snapshot.id)
        category = GroupCategoryFactory(iteration=iteration, name="Documents")

        # Map nodes to category
        GroupCategoryEntryFactory(
            folder_id=node1.id,
            group_id=category.id,  # type: ignore[attr-defined]
            iteration=iteration,
            processed_name="Document 1",
        )
        GroupCategoryEntryFactory(
            folder_id=node2.id,
            group_id=category.id,  # type: ignore[attr-defined]
            iteration=iteration,
            processed_name="Document 2",
        )
        storage_work_session.commit()

        hierarchy, items = create_hierarchy_from_categories(
            run.id,  # type: ignore[attr-defined]
            snapshot.id,  # type: ignore[attr-defined]
            PipelineStage.organized,
            storage_manager,
        )

        # Should have root, one category, and two nodes
        assert len(items) >= 4

        # Root should have one category child
        assert len(hierarchy.root.children) == 1
        category_record = hierarchy.root.children[0]
        assert category_record.name == "Documents"

        # Category should have two node children
        assert len(category_record.children) == 2
        node_names = {child.name for child in category_record.children}
        assert node_names == {"Document 1", "Document 2"}

    def test_uncategorized_nodes_pruned(
        self, storage_manager, storage_index_session, storage_work_session
    ):
        """TC-BUILD-06: Uncategorized nodes are absent from organized hierarchy."""
        snapshot = SnapshotFactory()
        categorized_node = NodeFactory(snapshot_id=snapshot.id, name="categorized.txt")
        uncategorized_node = NodeFactory(
            snapshot_id=snapshot.id, name="uncategorized.txt"
        )
        storage_index_session.commit()

        run = RunFactory(snapshot_id=snapshot.id)
        iteration = GroupIterationFactory(run=run, snapshot_id=snapshot.id)
        category = GroupCategoryFactory(iteration=iteration, name="Work")

        # Only categorize one node
        GroupCategoryEntryFactory(
            folder_id=categorized_node.id,
            group_id=category.id,  # type: ignore[attr-defined]
            iteration=iteration,
            processed_name="Work File",
        )
        storage_work_session.commit()

        # Build organized hierarchy only
        hierarchy, items = create_hierarchy_from_categories(
            run.id,  # type: ignore[attr-defined]
            snapshot.id,  # type: ignore[attr-defined]
            PipelineStage.organized,
            storage_manager,
        )

        # Uncategorized node should NOT be in items
        assert f"node-{uncategorized_node.id}" not in items

        # Categorized node should be in items
        assert f"node-{categorized_node.id}" in items


class TestBuildDualRepresentation:
    """Test build_dual_representation API-level function (TC-BUILD-07)."""

    def test_item_store_union(
        self, storage_manager, storage_index_session, storage_work_session
    ):
        """TC-BUILD-07: ItemStore is union of all requested hierarchies."""
        snapshot = SnapshotFactory()
        categorized_node = NodeFactory(snapshot_id=snapshot.id, name="categorized.txt")
        uncategorized_node = NodeFactory(
            snapshot_id=snapshot.id, name="uncategorized.txt"
        )
        storage_index_session.commit()

        run = RunFactory(snapshot_id=snapshot.id)
        iteration = GroupIterationFactory(run=run, snapshot_id=snapshot.id)
        category = GroupCategoryFactory(iteration=iteration, name="Work")

        # Only categorize one node
        GroupCategoryEntryFactory(
            folder_id=categorized_node.id,
            group_id=category.id,  # type: ignore[attr-defined]
            iteration=iteration,
            processed_name="Work File",
        )
        storage_work_session.commit()

        # Build both original and organized
        dual_rep = build_dual_representation(
            storage_manager,
            snapshot.id,  # type: ignore[attr-defined]
            run.id,  # type: ignore[attr-defined]
            stages=[PipelineStage.original, PipelineStage.organized],
        )

        # ItemStore should contain both categorized and uncategorized nodes
        # because uncategorized is in original hierarchy
        assert f"node-{categorized_node.id}" in dual_rep.items
        assert f"node-{uncategorized_node.id}" in dual_rep.items

        # Both hierarchies should exist
        assert "old" in dual_rep.hierarchies  # PipelineStage.original.value = "old"
        assert "new" in dual_rep.hierarchies  # PipelineStage.organized.value = "new"


class TestConvertHierarchyToFolderStructure:
    """Test convert_hierarchy_to_folder_structure function."""

    def test_conversion(self, storage_manager, storage_index_session):
        """Test converting hierarchy to FolderV2 format."""
        snapshot = SnapshotFactory()
        _ = NodeFactory(
            snapshot_id=snapshot.id,
            name="test.txt",
            kind=NodeKind.FILE,
            abs_path="/test/test.txt",
        )
        storage_index_session.commit()

        hierarchy, items = create_hierarchy_from_nodes(snapshot.id, storage_manager)  # type: ignore[attr-defined]

        # Convert to FolderV2 format
        folder_v2 = convert_hierarchy_to_folder_structure(hierarchy, items)

        # Verify structure - FolderV2 object
        assert isinstance(folder_v2, FolderV2)
        assert folder_v2.name == "root"

        # Find the child node
        assert len(folder_v2.children) == 1
        child = folder_v2.children[0]
        assert child.name == "test.txt"
        assert child.originalPath == "/test/test.txt"


class TestDataIntegrity:
    """Test data integrity and metadata (TC-BUILD-08)."""

    def test_originalPath_in_items(self, storage_manager, storage_index_session):
        """Verify originalPath is in HierarchyItem."""
        snapshot = SnapshotFactory()
        node = NodeFactory(
            snapshot_id=snapshot.id,
            name="file.txt",
            abs_path="/test/path/file.txt",
        )
        storage_index_session.commit()

        hierarchy, items = create_hierarchy_from_nodes(snapshot.id, storage_manager)  # type: ignore[attr-defined]

        node_item = items[f"node-{node.id}"]
        assert node_item.originalPath == "/test/path/file.txt"

    def test_name_in_record_not_item(self, storage_manager, storage_index_session):
        """Verify name is in HierarchyRecord, not HierarchyItem."""
        snapshot = SnapshotFactory()
        node = NodeFactory(
            snapshot_id=snapshot.id,
            name="testfile.txt",
        )
        storage_index_session.commit()

        hierarchy, items = create_hierarchy_from_nodes(snapshot.id, storage_manager)  # type: ignore[attr-defined]

        # Item should NOT have name
        node_item = items[f"node-{node.id}"]
        assert not hasattr(node_item, "name")

        # Record should have name
        node_record = hierarchy.root.children[0]
        assert node_record.name == "testfile.txt"

    def test_different_names_in_different_trees(
        self, storage_manager, storage_index_session, storage_work_session
    ):
        """
        Critical test: Same item can have different names in different hierarchies.

        This validates the core design requirement.
        """
        snapshot = SnapshotFactory()
        node = NodeFactory(
            snapshot_id=snapshot.id,
            name="original_name.txt",
        )
        storage_index_session.commit()

        run = RunFactory(snapshot_id=snapshot.id)
        iteration = GroupIterationFactory(run=run, snapshot_id=snapshot.id)
        category = GroupCategoryFactory(iteration=iteration, name="Work")

        # Use different name in category
        GroupCategoryEntryFactory(
            folder_id=node.id,
            group_id=category.id,  # type: ignore[attr-defined]
            iteration=iteration,
            processed_name="renamed_file.txt",  # Different name!
        )
        storage_work_session.commit()

        # Build both hierarchies
        dual_rep = build_dual_representation(
            storage_manager,
            snapshot.id,  # type: ignore[attr-defined]
            run.id,  # type: ignore[attr-defined]
            stages=[PipelineStage.original, PipelineStage.organized],
        )

        # Same item ID in both hierarchies
        node_id = f"node-{node.id}"
        assert node_id in dual_rep.items

        # Find the node in original hierarchy (key is "old")
        original_record = dual_rep.hierarchies["old"].root.children[0]
        assert original_record.itemId == node_id
        assert original_record.name == "original_name.txt"

        # Find the node in organized hierarchy (key is "new")
        organized_category = dual_rep.hierarchies["new"].root.children[0]
        organized_record = organized_category.children[0]
        assert organized_record.itemId == node_id
        assert organized_record.name == "renamed_file.txt"

        # CRITICAL: Same itemId, different names in different trees!
        assert original_record.name != organized_record.name
