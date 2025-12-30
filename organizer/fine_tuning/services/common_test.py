"""Tests for common.py"""

import pytest
from fine_tuning.services.common import (
    FeatureNodeCore,
    _load_and_index_nodes,
    _precompute_descendant_extensions,
    extract_feature_nodes,
    load_samples,
)
from storage.index_models import Node
from storage.manager import NodeKind
from storage.training_models import TrainingSample


class TestLoadSamples:
    """Test load_samples function"""

    def test_load_all_samples(self, training_session, label_run):
        """Test loading all samples without filters"""
        # Create test samples
        samples = [
            TrainingSample(
                label_run_id=label_run.id,
                snapshot_id=1,
                node_id=1,
                name_raw="test1",
                label="variant",
                split="train",
            ),
            TrainingSample(
                label_run_id=label_run.id,
                snapshot_id=1,
                node_id=2,
                name_raw="test2",
                label="subject",
                split="validation",
            ),
            TrainingSample(
                label_run_id=label_run.id,
                snapshot_id=1,
                node_id=3,
                name_raw="test3",
                label=None,
                split="test",
            ),
        ]
        training_session.add_all(samples)
        training_session.commit()

        # Load all samples
        loaded = load_samples(training_session)
        assert len(loaded) == 3

    @pytest.mark.parametrize(
        "split,expected_count",
        [
            ("train", 1),
            ("validation", 1),
            ("test", 1),
        ],
    )
    def test_load_samples_by_split(self, training_session, label_run, split, expected_count):
        """Test loading samples filtered by split"""
        samples = [
            TrainingSample(
                label_run_id=label_run.id,
                snapshot_id=1,
                node_id=1,
                name_raw="test1",
                label="variant",
                split="train",
            ),
            TrainingSample(
                label_run_id=label_run.id,
                snapshot_id=1,
                node_id=2,
                name_raw="test2",
                label="subject",
                split="validation",
            ),
            TrainingSample(
                label_run_id=label_run.id,
                snapshot_id=1,
                node_id=3,
                name_raw="test3",
                label="other",
                split="test",
            ),
        ]
        training_session.add_all(samples)
        training_session.commit()

        loaded = load_samples(training_session, split=split)
        assert len(loaded) == expected_count

    def test_load_labeled_only(self, training_session, label_run):
        """Test loading only labeled samples"""
        samples = [
            TrainingSample(
                label_run_id=label_run.id,
                snapshot_id=1,
                node_id=1,
                name_raw="test1",
                label="variant",
                split="train",
            ),
            TrainingSample(
                label_run_id=label_run.id,
                snapshot_id=1,
                node_id=2,
                name_raw="test2",
                label=None,
                split="train",
            ),
            TrainingSample(
                label_run_id=label_run.id,
                snapshot_id=1,
                node_id=3,
                name_raw="test3",
                label="",
                split="train",
            ),
        ]
        training_session.add_all(samples)
        training_session.commit()

        loaded = load_samples(training_session, labeled_only=True)
        assert len(loaded) == 1
        assert loaded[0].label == "variant"

    def test_load_by_label_run_id(self, training_session, label_run):
        """Test loading samples filtered by label run ID"""
        # Create another label run
        label_run2 = LabelRun(snapshot_id=2, label_source="test2")
        training_session.add(label_run2)
        training_session.flush()

        samples = [
            TrainingSample(
                label_run_id=label_run.id,
                snapshot_id=1,
                node_id=1,
                name_raw="test1",
                label="variant",
            ),
            TrainingSample(
                label_run_id=label_run2.id,
                snapshot_id=2,
                node_id=2,
                name_raw="test2",
                label="subject",
            ),
        ]
        training_session.add_all(samples)
        training_session.commit()

        loaded = load_samples(training_session, label_run_id=label_run.id)
        assert len(loaded) == 1
        assert loaded[0].label == "variant"

    def test_load_combined_filters(self, training_session, label_run):
        """Test loading samples with multiple filters combined"""
        samples = [
            TrainingSample(
                label_run_id=label_run.id,
                snapshot_id=1,
                node_id=1,
                name_raw="test1",
                label="variant",
                split="train",
            ),
            TrainingSample(
                label_run_id=label_run.id,
                snapshot_id=1,
                node_id=2,
                name_raw="test2",
                label="subject",
                split="validation",
            ),
            TrainingSample(
                label_run_id=label_run.id,
                snapshot_id=1,
                node_id=3,
                name_raw="test3",
                label=None,
                split="train",
            ),
        ]
        training_session.add_all(samples)
        training_session.commit()

        loaded = load_samples(
            training_session,
            split="train",
            labeled_only=True,
            label_run_id=label_run.id,
        )
        assert len(loaded) == 1
        assert loaded[0].label == "variant"


class TestLoadAndIndexNodes:
    """Test _load_and_index_nodes function"""

    def test_load_basic_hierarchy(self, index_session, sample_snapshot):
        """Test loading nodes and building indexes"""
        nodes = [
            Node(
                node_id=1,
                snapshot_id=1,
                name="root",
                kind=NodeKind.DIR,
                parent_node_id=None,
                depth=0,
                rel_path="root",
                abs_path="/test/root",
                ext=None,
                file_source="filesystem",
            ),
            Node(
                node_id=2,
                snapshot_id=1,
                name="child",
                kind=NodeKind.DIR,
                parent_node_id=1,
                depth=1,
                rel_path="root/child",
                abs_path="/test/root/child",
                ext=None,
                file_source="filesystem",
            ),
            Node(
                node_id=3,
                snapshot_id=1,
                name="file.txt",
                kind=NodeKind.FILE,
                parent_node_id=2,
                depth=2,
                rel_path="root/child/file.txt",
                abs_path="/test/root/child/file.txt",
                ext="txt",
                file_source="filesystem",
            ),
        ]
        index_session.add_all(nodes)
        index_session.commit()

        nodes_by_id, processed_name_by_id, children_by_parent = _load_and_index_nodes(
            index_session, 1
        )

        assert len(nodes_by_id) == 3
        assert 1 in nodes_by_id
        assert 2 in nodes_by_id
        assert 3 in nodes_by_id

        assert len(processed_name_by_id) == 3
        assert processed_name_by_id[1] == "root"

        assert len(children_by_parent) == 2
        assert 2 in children_by_parent[1]
        assert 3 in children_by_parent[2]

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
        # Create mock nodes
        nodes_by_id = {
            1: Node(
                node_id=1,
                snapshot_id=1,
                name="root",
                kind=NodeKind.DIR,
                parent_node_id=None,
                depth=0,
                rel_path="root",
                abs_path="/test/root",
                ext=None,
                file_source="filesystem",
            ),
            2: Node(
                node_id=2,
                snapshot_id=1,
                name="file.txt",
                kind=NodeKind.FILE,
                parent_node_id=1,
                depth=1,
                rel_path="root/file.txt",
                abs_path="/test/root/file.txt",
                ext=".txt",
                file_source="filesystem",
            ),
            3: Node(
                node_id=3,
                snapshot_id=1,
                name="file.png",
                kind=NodeKind.FILE,
                parent_node_id=1,
                depth=1,
                rel_path="root/file.png",
                abs_path="/test/root/file.png",
                ext=".png",
                file_source="filesystem",
            ),
        }

        children_by_parent = {
            None: [1],
            1: [2, 3],
        }

        result = _precompute_descendant_extensions(nodes_by_id, children_by_parent)

        assert len(result) == 3
        assert "txt" in result[1]
        assert "png" in result[1]
        assert "txt" in result[2]
        assert "png" in result[3]

    def test_nested_hierarchy(self):
        """Test extension computation with nested folders"""
        nodes_by_id = {
            1: Node(
                node_id=1,
                snapshot_id=1,
                name="root",
                kind=NodeKind.DIR,
                parent_node_id=None,
                depth=0,
                rel_path="root",
                abs_path="/test/root",
                ext=None,
                file_source="filesystem",
            ),
            2: Node(
                node_id=2,
                snapshot_id=1,
                name="subfolder",
                kind=NodeKind.DIR,
                parent_node_id=1,
                depth=1,
                rel_path="root/subfolder",
                abs_path="/test/root/subfolder",
                ext=None,
                file_source="filesystem",
            ),
            3: Node(
                node_id=3,
                snapshot_id=1,
                name="deep.jpg",
                kind=NodeKind.FILE,
                parent_node_id=2,
                depth=2,
                rel_path="root/subfolder/deep.jpg",
                abs_path="/test/root/subfolder/deep.jpg",
                ext=".jpg",
                file_source="filesystem",
            ),
        }

        children_by_parent = {
            None: [1],
            1: [2],
            2: [3],
        }

        result = _precompute_descendant_extensions(nodes_by_id, children_by_parent)

        # Root should have jpg from deep descendant
        assert "jpg" in result[1]
        # Subfolder should have jpg from direct child
        assert "jpg" in result[2]
        # File should have its own extension
        assert "jpg" in result[3]


class TestExtractFeatureNodes:
    """Test extract_feature_nodes function"""

    def test_basic_extraction(self, index_session, sample_snapshot):
        """Test basic feature node extraction"""
        nodes = [
            Node(
                node_id=1,
                snapshot_id=1,
                name="Parent",
                kind=NodeKind.DIR,
                parent_node_id=None,
                depth=0,
                rel_path="Parent",
                abs_path="/test/Parent",
                ext=None,
                file_source="filesystem",
            ),
            Node(
                node_id=2,
                snapshot_id=1,
                name="Target",
                kind=NodeKind.DIR,
                parent_node_id=1,
                depth=1,
                rel_path="Parent/Target",
                abs_path="/test/Parent/Target",
                ext=None,
                file_source="filesystem",
            ),
            Node(
                node_id=3,
                snapshot_id=1,
                name="file.txt",
                kind=NodeKind.FILE,
                parent_node_id=2,
                depth=2,
                rel_path="Parent/Target/file.txt",
                abs_path="/test/Parent/Target/file.txt",
                ext=".txt",
                file_source="filesystem",
            ),
        ]
        index_session.add_all(nodes)
        index_session.commit()

        feature_nodes = extract_feature_nodes(
            index_session=index_session,
            snapshot_id=1,
            nodes=[nodes[1]],  # Extract for "Target" node
            max_siblings=5,
            max_descendents=10,
            max_children=5,
        )

        assert len(feature_nodes) == 1
        feature_node = feature_nodes[0]

        assert feature_node.node.node_id == 2
        assert feature_node.parent.node_id == 1
        assert feature_node.grandparent is None
        assert len(feature_node.child_nodes) == 1
        assert len(feature_node.sibling_nodes) == 0
        assert "txt" in feature_node.descendent_extentions

    def test_with_siblings(self, index_session, sample_snapshot):
        """Test extraction with sibling nodes"""
        nodes = [
            Node(
                node_id=1,
                snapshot_id=1,
                name="Parent",
                kind=NodeKind.DIR,
                parent_node_id=None,
                depth=0,
                rel_path="Parent",
                abs_path="/test/Parent",
                ext=None,
                file_source="filesystem",
            ),
            Node(
                node_id=2,
                snapshot_id=1,
                name="Target",
                kind=NodeKind.DIR,
                parent_node_id=1,
                depth=1,
                rel_path="Parent/Target",
                abs_path="/test/Parent/Target",
                ext=None,
                file_source="filesystem",
            ),
            Node(
                node_id=3,
                snapshot_id=1,
                name="Sibling1",
                kind=NodeKind.DIR,
                parent_node_id=1,
                depth=1,
                rel_path="Parent/Sibling1",
                abs_path="/test/Parent/Sibling1",
                ext=None,
                file_source="filesystem",
            ),
            Node(
                node_id=4,
                snapshot_id=1,
                name="Sibling2",
                kind=NodeKind.DIR,
                parent_node_id=1,
                depth=1,
                rel_path="Parent/Sibling2",
                abs_path="/test/Parent/Sibling2",
                ext=None,
                file_source="filesystem",
            ),
        ]
        index_session.add_all(nodes)
        index_session.commit()

        feature_nodes = extract_feature_nodes(
            index_session=index_session,
            snapshot_id=1,
            nodes=[nodes[1]],
            max_siblings=5,
            max_descendents=10,
            max_children=5,
        )

        assert len(feature_nodes) == 1
        feature_node = feature_nodes[0]

        assert len(feature_node.sibling_nodes) == 2

    def test_max_siblings_cap(self, index_session, sample_snapshot):
        """Test that max_siblings limits sibling count"""
        parent = Node(
            node_id=1,
            snapshot_id=1,
            name="Parent",
            kind=NodeKind.DIR,
            parent_node_id=None,
            depth=0,
            rel_path="Parent",
            abs_path="/test/Parent",
            ext=None,
            file_source="filesystem",
        )
        index_session.add(parent)

        target = Node(
            node_id=2,
            snapshot_id=1,
            name="Target",
            kind=NodeKind.DIR,
            parent_node_id=1,
            depth=1,
            rel_path="Parent/Target",
            abs_path="/test/Parent/Target",
            ext=None,
            file_source="filesystem",
        )
        index_session.add(target)

        # Add 10 siblings
        for i in range(10):
            sibling = Node(
                node_id=100 + i,
                snapshot_id=1,
                name=f"Sibling{i}",
                kind=NodeKind.DIR,
                parent_node_id=1,
                depth=1,
                rel_path=f"Parent/Sibling{i}",
                abs_path=f"/test/Parent/Sibling{i}",
                ext=None,
                file_source="filesystem",
            )
            index_session.add(sibling)

        index_session.commit()

        feature_nodes = extract_feature_nodes(
            index_session=index_session,
            snapshot_id=1,
            nodes=[target],
            max_siblings=3,
            max_descendents=10,
            max_children=5,
        )

        assert len(feature_nodes) == 1
        # Should cap at max_siblings
        assert len(feature_nodes[0].sibling_nodes) <= 3

    def test_zip_file_extraction(self, index_session, sample_snapshot):
        """Test extraction for ZIP files"""
        zip_node = Node(
            node_id=1,
            snapshot_id=1,
            name="archive.zip",
            kind=NodeKind.FILE,
            parent_node_id=None,
            depth=0,
            rel_path="archive.zip",
            abs_path="/test/archive.zip",
            ext=".zip",
            file_source="zip_file",
        )
        index_session.add(zip_node)
        index_session.commit()

        feature_nodes = extract_feature_nodes(
            index_session=index_session,
            snapshot_id=1,
            nodes=[zip_node],
            max_siblings=5,
            max_descendents=10,
            max_children=5,
        )

        # ZIP files should be extracted
        assert len(feature_nodes) == 1
        assert feature_nodes[0].node.ext == ".zip"

    def test_skip_regular_files(self, index_session, sample_snapshot):
        """Test that regular files are skipped"""
        regular_file = Node(
            node_id=1,
            snapshot_id=1,
            name="file.txt",
            kind=NodeKind.FILE,
            parent_node_id=None,
            depth=0,
            rel_path="file.txt",
            abs_path="/test/file.txt",
            ext=".txt",
            file_source="filesystem",
        )
        index_session.add(regular_file)
        index_session.commit()

        feature_nodes = extract_feature_nodes(
            index_session=index_session,
            snapshot_id=1,
            nodes=[regular_file],
            max_siblings=5,
            max_descendents=10,
            max_children=5,
        )

        # Regular files should be skipped
        assert len(feature_nodes) == 0


class TestFeatureNodeCore:
    """Test FeatureNodeCore model"""

    def test_property_access(self, index_session, sample_snapshot):
        """Test that property accessors work correctly"""
        grandparent = Node(
            node_id=1,
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
            node_id=2,
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
            node_id=3,
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
            node_id=4,
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
            node_id=5,
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
            node_id=1,
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
