"""Tests for common.py"""

import pytest
from storage.factories import (
    LabelRunFactory,
    NodeFactory,
    TrainingSampleFactory,
)
from storage.index_models import Node
from storage.manager import NodeKind

from fine_tuning.services.common import (
    FeatureNodeCore,
    _load_and_index_nodes,
    _precompute_descendant_extensions,
    extract_feature_nodes,
    load_samples,
)


class TestLoadSamples:
    """Test load_samples function"""

    def test_load_all_samples(self, training_session, label_run):
        """Test loading all samples without filters"""
        created_samples = TrainingSampleFactory.create_batch(2, label_run=label_run)
        created_samples.append(TrainingSampleFactory(label_run=label_run, label=None))

        loaded = load_samples(training_session)

        # Verify count and that all created samples are present
        assert len(loaded) == 3
        assert {s.id for s in loaded} == {s.id for s in created_samples}

    @pytest.mark.parametrize(
        "split,expected_count",
        [
            ("train", 1),
            ("validation", 1),
            ("test", 1),
        ],
    )
    def test_load_samples_by_split(
        self, training_session, label_run, split, expected_count
    ):
        """Test loading samples filtered by split"""
        TrainingSampleFactory(label_run=label_run)
        TrainingSampleFactory(label_run=label_run, split="validation")
        TrainingSampleFactory(label_run=label_run, split="test")

        loaded = load_samples(training_session, split=split)
        assert len(loaded) == expected_count
        if expected_count > 0:
            assert all(sample.split == split for sample in loaded)

    def test_load_labeled_only(self, training_session, label_run):
        """Test loading only labeled samples"""
        TrainingSampleFactory(label_run=label_run, label="asset_type")
        TrainingSampleFactory(label_run=label_run, label=None)
        TrainingSampleFactory(label_run=label_run, label="")

        loaded = load_samples(training_session, labeled_only=True)
        assert len(loaded) == 1
        assert loaded[0].label == "asset_type"

    def test_load_by_label_run_id(self, training_session, label_run):
        """Test loading samples filtered by label run ID"""
        label_run2 = LabelRunFactory(snapshot_id=2)

        TrainingSampleFactory(label_run=label_run, label="asset_type")
        TrainingSampleFactory(label_run=label_run2, label="content_subject")  # type: ignore[attr-defined]

        loaded = load_samples(training_session, label_run_id=label_run.id)
        assert len(loaded) == 1
        assert loaded[0].label == "asset_type"

    def test_load_combined_filters(self, training_session, label_run):
        """Test loading samples with multiple filters combined"""
        TrainingSampleFactory(label_run=label_run, label="asset_type", split="train")
        TrainingSampleFactory(
            label_run=label_run, label="content_subject", split="validation"
        )
        TrainingSampleFactory(label_run=label_run, label=None)

        loaded = load_samples(
            training_session,
            split="train",
            labeled_only=True,
            label_run_id=label_run.id,
        )
        assert len(loaded) == 1
        assert loaded[0].label == "asset_type"


class TestLoadAndIndexNodes:
    """Test _load_and_index_nodes function"""

    def test_load_basic_hierarchy(self, index_session, sample_snapshot):
        """Test loading nodes and building indexes"""
        snapshot_id = sample_snapshot.id
        NodeFactory(
            id=1,
            snapshot_id=snapshot_id,
            name="root",
            rel_path="root",
            abs_path="/test/root",
            parent_node_id=None,
            depth=0,
        )
        NodeFactory(
            id=2,
            snapshot_id=snapshot_id,
            name="child",
            rel_path="root/child",
            abs_path="/test/root/child",
            parent_node_id=1,
            depth=1,
        )
        NodeFactory(
            id=3,
            snapshot_id=snapshot_id,
            name="file.txt",
            rel_path="root/child/file.txt",
            abs_path="/test/root/child/file.txt",
            kind=NodeKind.FILE,
            parent_node_id=2,
            depth=2,
            ext="txt",
        )

        nodes_by_id, processed_name_by_id, children_by_parent = _load_and_index_nodes(
            index_session, snapshot_id
        )

        assert set(nodes_by_id.keys()) == {1, 2, 3}
        assert set(processed_name_by_id.keys()) == {1, 2, 3}
        assert processed_name_by_id[1] == "root"

        assert set(children_by_parent.keys()) == {None, 1, 2}
        assert children_by_parent[None] == [1]
        assert children_by_parent[1] == [2]
        assert children_by_parent[2] == [3]

    def test_empty_snapshot(self, index_session, sample_snapshot):
        """Test loading from empty snapshot"""
        nodes_by_id, processed_name_by_id, children_by_parent = _load_and_index_nodes(
            index_session, 1
        )

        assert len(nodes_by_id) == 0
        assert len(processed_name_by_id) == 0
        assert len(children_by_parent) == 0


class TestPrecomputeDescendantExtensions:
    """Test _precompute_descendant_extensions function"""

    def test_basic_extension_computation(self):
        """Test computing descendant extensions"""
        # Use factory.build() to create nodes without persisting to DB
        nodes_by_id = {
            1: NodeFactory.build(
                id=1,
                kind=NodeKind.DIR,
                parent_node_id=None,
                ext=None,
                depth=0,
            ),
            2: NodeFactory.build(
                id=2,
                kind=NodeKind.FILE,
                parent_node_id=1,
                ext=".txt",
                depth=1,
            ),
            3: NodeFactory.build(
                id=3,
                kind=NodeKind.FILE,
                parent_node_id=1,
                ext=".png",
                depth=1,
            ),
        }

        children_by_parent = {
            None: [1],
            1: [2, 3],
        }

        result = _precompute_descendant_extensions(nodes_by_id, children_by_parent)

        assert result[1] == {"txt", "png"}
        assert result[2] == {"txt"}
        assert result[3] == {"png"}
        assert set(result.keys()) == {1, 2, 3}

    def test_nested_hierarchy(self):
        """Test extension computation with nested folders"""
        nodes_by_id = {
            1: NodeFactory.build(
                id=1,
                kind=NodeKind.DIR,
                parent_node_id=None,
                ext=None,
                depth=0,
            ),
            2: NodeFactory.build(
                id=2,
                kind=NodeKind.DIR,
                parent_node_id=1,
                ext=None,
                depth=1,
            ),
            3: NodeFactory.build(
                id=3,
                kind=NodeKind.FILE,
                parent_node_id=2,
                ext=".jpg",
                depth=2,
            ),
        }

        children_by_parent = {
            None: [1],
            1: [2],
            2: [3],
        }

        result = _precompute_descendant_extensions(nodes_by_id, children_by_parent)

        # Root should have jpg from deep descendant
        assert result[1] == {"jpg"}
        # Subfolder should have jpg from direct child
        assert result[2] == {"jpg"}
        # File should have its own extension
        assert result[3] == {"jpg"}


class TestExtractFeatureNodes:
    """Test extract_feature_nodes function"""

    def test_basic_extraction(self, index_session, sample_snapshot):
        """Test basic feature node extraction"""
        snapshot_id = sample_snapshot.id
        _parent = NodeFactory(
            id=1,
            snapshot_id=snapshot_id,
            name="Parent",
            rel_path="Parent",
            abs_path="/test/Parent",
            depth=0,
        )
        target = NodeFactory(
            id=2,
            snapshot_id=snapshot_id,
            name="Target",
            rel_path="Parent/Target",
            abs_path="/test/Parent/Target",
            parent_node_id=1,
            depth=1,
        )
        NodeFactory(
            id=3,
            snapshot_id=snapshot_id,
            name="file.txt",
            rel_path="Parent/Target/file.txt",
            abs_path="/test/Parent/Target/file.txt",
            kind=NodeKind.FILE,
            parent_node_id=2,
            depth=2,
            ext=".txt",
        )

        feature_nodes = extract_feature_nodes(
            index_session=index_session,
            snapshot_id=snapshot_id,
            nodes=[target],  # type: ignore[list-item]
            max_siblings=5,
            max_descendents=10,
            max_children=5,
        )

        assert len(feature_nodes) == 1
        feature_node = feature_nodes[0]

        assert feature_node.node.id == 2
        assert feature_node.parent.id == 1
        assert feature_node.grandparent is None
        assert len(feature_node.child_nodes) == 1
        assert len(feature_node.sibling_nodes) == 0
        assert feature_node.descendent_extentions == ["txt"]

    def test_with_siblings(self, index_session, sample_snapshot):
        """Test extraction with sibling nodes"""
        snapshot_id = sample_snapshot.id
        _parent = NodeFactory(id=1, snapshot_id=snapshot_id, name="Parent", depth=0)
        target = NodeFactory(
            id=2,
            snapshot_id=snapshot_id,
            name="Target",
            parent_node_id=1,
            depth=1,
        )
        NodeFactory(
            id=3,
            snapshot_id=snapshot_id,
            name="Sibling1",
            parent_node_id=1,
            depth=1,
        )
        NodeFactory(
            id=4,
            snapshot_id=snapshot_id,
            name="Sibling2",
            parent_node_id=1,
            depth=1,
        )

        feature_nodes = extract_feature_nodes(
            index_session=index_session,
            snapshot_id=snapshot_id,
            nodes=[target],  # type: ignore[list-item]
            max_siblings=5,
            max_descendents=10,
            max_children=5,
        )

        assert len(feature_nodes) == 1
        feature_node = feature_nodes[0]

        assert {node.name for node in feature_node.sibling_nodes} == {
            "Sibling1",
            "Sibling2",
        }

    def test_max_siblings_cap(self, index_session, sample_snapshot):
        """Test that max_siblings limits sibling count"""
        snapshot_id = sample_snapshot.id
        _parent = NodeFactory(id=1, snapshot_id=snapshot_id, name="Parent", depth=0)
        target = NodeFactory(
            id=2,
            snapshot_id=snapshot_id,
            name="Target",
            parent_node_id=1,
            depth=1,
        )

        # Add 10 siblings using factory
        for i in range(10):
            NodeFactory(
                id=100 + i,
                snapshot_id=snapshot_id,
                name=f"Sibling{i}",
                parent_node_id=1,
                depth=1,
            )

        feature_nodes = extract_feature_nodes(
            index_session=index_session,
            snapshot_id=snapshot_id,
            nodes=[target],  # type: ignore[list-item]
            max_siblings=3,
            max_descendents=10,
            max_children=5,
        )

        assert len(feature_nodes) == 1
        # Should cap at max_siblings
        assert len(feature_nodes[0].sibling_nodes) == 3

    def test_zip_file_extraction(self, index_session, sample_snapshot):
        """Test extraction for ZIP files"""
        snapshot_id = sample_snapshot.id
        zip_node = NodeFactory(
            id=1,
            snapshot_id=snapshot_id,
            name="archive.zip",
            kind=NodeKind.FILE,
            ext=".zip",
            file_source="zip_file",
            depth=0,
        )

        feature_nodes = extract_feature_nodes(
            index_session=index_session,
            snapshot_id=snapshot_id,
            nodes=[zip_node],  # type: ignore[list-item]
            max_siblings=5,
            max_descendents=10,
            max_children=5,
        )

        assert len(feature_nodes) == 1
        assert feature_nodes[0].node.ext == ".zip"

    def test_skip_regular_files(self, index_session, sample_snapshot):
        """Test that regular files are skipped"""
        snapshot_id = sample_snapshot.id
        regular_file = NodeFactory(
            id=1,
            snapshot_id=snapshot_id,
            name="file.txt",
            kind=NodeKind.FILE,
            ext=".txt",
            depth=0,
        )

        feature_nodes = extract_feature_nodes(
            index_session=index_session,
            snapshot_id=snapshot_id,
            nodes=[regular_file],  # type: ignore[list-item]
            max_siblings=5,
            max_descendents=10,
            max_children=5,
        )

        # Regular files should be skipped
        assert len(feature_nodes) == 0


class TestFeatureNodeCore:
    """Test FeatureNodeCore model"""

    def test_property_access(self):
        """Test that property accessors work correctly"""
        # Use factories to create nodes
        grandparent = Node(
            id=1,
            snapshot_id=1,
            name="Grandparent",
            kind=NodeKind.DIR,
            parent_node_id=None,
            depth=0,
            rel_path="Grandparent",
            abs_path="/test/Grandparent",
            ext=None,
            file_source="filesystem",
        )
        parent = Node(
            id=2,
            snapshot_id=1,
            name="Parent",
            kind=NodeKind.DIR,
            parent_node_id=1,
            depth=1,
            rel_path="Grandparent/Parent",
            abs_path="/test/Grandparent/Parent",
            ext=None,
            file_source="filesystem",
        )
        node = Node(
            id=3,
            snapshot_id=1,
            name="Target",
            kind=NodeKind.DIR,
            parent_node_id=2,
            depth=2,
            rel_path="Grandparent/Parent/Target",
            abs_path="/test/Grandparent/Parent/Target",
            ext=None,
            file_source="filesystem",
        )
        child = Node(
            id=4,
            snapshot_id=1,
            name="Child",
            kind=NodeKind.DIR,
            parent_node_id=3,
            depth=3,
            rel_path="Grandparent/Parent/Target/Child",
            abs_path="/test/Grandparent/Parent/Target/Child",
            ext=None,
            file_source="filesystem",
        )
        sibling = Node(
            id=5,
            snapshot_id=1,
            name="Sibling",
            kind=NodeKind.DIR,
            parent_node_id=2,
            depth=2,
            rel_path="Grandparent/Parent/Sibling",
            abs_path="/test/Grandparent/Parent/Sibling",
            ext=None,
            file_source="filesystem",
        )

        feature_node = FeatureNodeCore(
            snapshot_id=1,
            node=node,
            parent=parent,
            grandparent=grandparent,
            child_nodes=[child],
            sibling_nodes=[sibling],
            descendent_extentions=["txt", "png"],
            max_siblings=5,
            max_descendents=10,
            max_children=5,
        )

        assert feature_node.grandparent_name == "Grandparent"
        assert feature_node.parent_name == "Parent"
        assert feature_node.child_names == ["Child"]
        assert feature_node.sibling_names == ["Sibling"]

    def test_none_parent_grandparent(self):
        """Test properties when parent/grandparent are None"""
        node = Node(
            id=1,
            snapshot_id=1,
            name="Root",
            kind=NodeKind.DIR,
            parent_node_id=None,
            depth=0,
            rel_path="Root",
            abs_path="/test/Root",
            ext=None,
            file_source="filesystem",
        )

        feature_node = FeatureNodeCore(
            snapshot_id=1,
            node=node,
            parent=None,
            grandparent=None,
            child_nodes=[],
            sibling_nodes=[],
            descendent_extentions=[],
            max_siblings=5,
            max_descendents=10,
            max_children=5,
        )

        assert feature_node.grandparent_name is None
        assert feature_node.parent_name is None
