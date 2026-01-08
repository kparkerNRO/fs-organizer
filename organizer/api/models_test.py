"""Tests for API models, specifically the dual representation models."""

import pytest
from pydantic import ValidationError

from api.models import (
    DualRepresentation,
    Hierarchy,
    HierarchyDiff,
    HierarchyItem,
    HierarchyRecord,
)


class TestHierarchyItem:
    """Test HierarchyItem model validation."""

    def test_node_creation(self):
        """Test creating a node HierarchyItem."""
        item = HierarchyItem(
            id="node-123",
            name="Documents",
            type="node",
            originalPath="/home/user/Documents",
        )
        assert item.id == "node-123"
        assert item.name == "Documents"
        assert item.type == "node"
        assert item.originalPath == "/home/user/Documents"

    def test_category_creation(self):
        """Test creating a category HierarchyItem."""
        item = HierarchyItem(
            id="category-456",
            name="Personal Files",
            type="category",
        )
        assert item.id == "category-456"
        assert item.name == "Personal Files"
        assert item.type == "category"
        assert item.originalPath is None

    def test_invalid_type(self):
        """Test that invalid type raises validation error."""
        with pytest.raises(ValidationError):
            HierarchyItem(
                id="invalid-123",
                name="Test",
                type="invalid",  # type: ignore[arg-type]
            )

    def test_missing_required_fields(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            HierarchyItem(id="test-123", name="Test")  # type: ignore[call-arg]

    def test_optional_path_for_node(self):
        """Test that originalPath is optional even for nodes."""
        item = HierarchyItem(
            id="node-789",
            name="File",
            type="node",
        )
        assert item.originalPath is None


class TestHierarchyRecord:
    """Test HierarchyRecord model validation."""

    def test_simple_record(self):
        """Test creating a simple HierarchyRecord."""
        record = HierarchyRecord(id="node-1")
        assert record.id == "node-1"
        assert record.children == []
        assert record.metadata == {}

    def test_record_with_children(self):
        """Test creating a HierarchyRecord with children."""
        child1 = HierarchyRecord(id="child-1")
        child2 = HierarchyRecord(id="child-2")
        parent = HierarchyRecord(id="parent", children=[child1, child2])

        assert parent.id == "parent"
        assert len(parent.children) == 2
        assert parent.children[0].id == "child-1"
        assert parent.children[1].id == "child-2"

    def test_nested_hierarchy(self):
        """Test creating a deeply nested hierarchy."""
        grandchild = HierarchyRecord(id="grandchild")
        child = HierarchyRecord(id="child", children=[grandchild])
        root = HierarchyRecord(id="root", children=[child])

        assert root.id == "root"
        assert root.children[0].id == "child"
        assert root.children[0].children[0].id == "grandchild"

    def test_metadata(self):
        """Test HierarchyRecord with metadata."""
        record = HierarchyRecord(
            id="node-1", metadata={"custom_field": "value", "count": 5}
        )
        assert record.metadata["custom_field"] == "value"
        assert record.metadata["count"] == 5


class TestHierarchy:
    """Test Hierarchy model validation."""

    def test_hierarchy_creation(self):
        """Test creating a Hierarchy object."""
        root = HierarchyRecord(
            id="root",
            children=[
                HierarchyRecord(id="child1"),
                HierarchyRecord(id="child2"),
            ],
        )
        hierarchy = Hierarchy(
            stage="original",
            source_type="node",
            root=root,
        )
        assert hierarchy.stage == "original"
        assert hierarchy.source_type == "node"
        assert hierarchy.root.id == "root"
        assert len(hierarchy.root.children) == 2


class TestDualRepresentation:
    """Test DualRepresentation model validation."""

    def test_empty_dual_representation(self):
        """Test creating an empty dual representation."""
        dual_rep = DualRepresentation(
            items={},
            hierarchies={},
        )
        assert dual_rep.items == {}
        assert dual_rep.hierarchies == {}

    def test_dual_representation_with_data(self):
        """Test creating a dual representation with actual data."""
        items = {
            "node-1": HierarchyItem(
                id="node-1", name="root", type="node", originalPath="/"
            ),
            "node-2": HierarchyItem(
                id="node-2", name="docs", type="node", originalPath="/docs"
            ),
            "category-1": HierarchyItem(id="category-1", name="Work", type="category"),
        }
        hierarchies = {
            "original": Hierarchy(
                stage="original",
                source_type="node",
                root=HierarchyRecord(
                    id="node-1", children=[HierarchyRecord(id="node-2")]
                ),
            ),
            "organized": Hierarchy(
                stage="organized",
                source_type="category",
                root=HierarchyRecord(
                    id="category-1", children=[HierarchyRecord(id="node-2")]
                ),
            ),
        }

        dual_rep = DualRepresentation(
            items=items,
            hierarchies=hierarchies,
        )

        assert len(dual_rep.items) == 3
        assert "node-1" in dual_rep.items
        assert "category-1" in dual_rep.items
        assert len(dual_rep.hierarchies) == 2
        assert "original" in dual_rep.hierarchies
        assert "organized" in dual_rep.hierarchies

    def test_model_serialization(self):
        """Test that model can be serialized to dict."""
        items = {
            "node-1": HierarchyItem(id="node-1", name="root", type="node"),
        }
        hierarchies = {
            "original": Hierarchy(
                stage="original",
                source_type="node",
                root=HierarchyRecord(id="node-1"),
            ),
        }
        dual_rep = DualRepresentation(
            items=items,
            hierarchies=hierarchies,
        )

        data = dual_rep.model_dump()
        assert "items" in data
        assert "hierarchies" in data
        assert data["items"]["node-1"]["name"] == "root"
        assert data["hierarchies"]["original"]["stage"] == "original"
        assert data["hierarchies"]["original"]["root"]["id"] == "node-1"


class TestHierarchyDiff:
    """Test HierarchyDiff model validation."""

    def test_empty_diff(self):
        """Test creating an empty diff."""
        diff = HierarchyDiff(added={}, deleted={})
        assert diff.added == {}
        assert diff.deleted == {}

    def test_diff_with_additions(self):
        """Test creating a diff with only additions."""
        diff = HierarchyDiff(
            added={"category-1": ["node-1", "node-2"]},
            deleted={},
        )
        assert diff.added["category-1"] == ["node-1", "node-2"]
        assert diff.deleted == {}

    def test_diff_with_deletions(self):
        """Test creating a diff with only deletions."""
        diff = HierarchyDiff(
            added={},
            deleted={"category-2": ["node-3"]},
        )
        assert diff.added == {}
        assert diff.deleted["category-2"] == ["node-3"]

    def test_diff_with_both_operations(self):
        """Test creating a diff with both additions and deletions."""
        diff = HierarchyDiff(
            added={"category-1": ["node-1", "node-2"]},
            deleted={"category-2": ["node-3", "node-4"]},
        )
        assert len(diff.added) == 1
        assert len(diff.deleted) == 1
        assert "node-1" in diff.added["category-1"]
        assert "node-3" in diff.deleted["category-2"]

    def test_diff_serialization(self):
        """Test that diff can be serialized to dict."""
        diff = HierarchyDiff(
            added={"category-1": ["node-1"]},
            deleted={"category-2": ["node-2"]},
        )
        data = diff.model_dump()
        assert "added" in data
        assert "deleted" in data
        assert data["added"]["category-1"] == ["node-1"]
        assert data["deleted"]["category-2"] == ["node-2"]

    def test_multiple_parents_in_diff(self):
        """Test diff with multiple parent categories."""
        diff = HierarchyDiff(
            added={
                "category-1": ["node-1", "node-2"],
                "category-2": ["node-3"],
                "category-3": ["node-4", "node-5", "node-6"],
            },
            deleted={
                "category-4": ["node-7"],
            },
        )
        assert len(diff.added) == 3
        assert len(diff.deleted) == 1
        assert len(diff.added["category-3"]) == 3
