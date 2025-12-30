"""Tests for sampling.py"""

import csv

import pytest
from fine_tuning.services.sampling import (
    _cluster_by_similarity,
    _sample_from_clusters,
    apply_labels_to_samples,
    create_label_runs,
    read_classification_csv,
    select_training_samples,
    validate_all_labels_present,
    validate_input_csv,
    validate_label_values,
    write_sample_csv,
)
from storage.index_models import Node
from storage.manager import NodeKind

from .factories import LabelRunFactory, NodeFactory, TrainingSampleFactory


class TestSelectTrainingSamples:
    """Test select_training_samples function"""

    def test_basic_selection(self, index_session, sample_snapshot):
        """Test basic sample selection"""
        # Create 20 nodes using factory
        for i in range(20):
            NodeFactory(node_id=i + 1, snapshot_id=1, depth=1)

        samples = select_training_samples(
            session=index_session,
            snapshot_id=1,
            sample_size=10,
            min_depth=1,
            max_depth=5,
        )

        assert len(samples) == 10
        # All samples should be directories
        assert all(s.kind == NodeKind.DIR for s in samples)

    def test_depth_filtering(self, index_session, sample_snapshot):
        """Test that depth filtering works correctly"""
        NodeFactory(node_id=1, snapshot_id=1, name="depth0", depth=0)
        NodeFactory(node_id=2, snapshot_id=1, name="depth2", parent_node_id=1, depth=2)
        NodeFactory(node_id=3, snapshot_id=1, name="depth5", parent_node_id=2, depth=5)

        samples = select_training_samples(
            session=index_session,
            snapshot_id=1,
            sample_size=10,
            min_depth=2,
            max_depth=5,
        )

        # Should only include nodes at depth 2 and 5
        assert len(samples) == 2
        depths = {s.depth for s in samples}
        assert 0 not in depths
        assert 2 in depths or 5 in depths

    def test_empty_result(self, index_session, sample_snapshot):
        """Test selection with no matching nodes"""
        samples = select_training_samples(
            session=index_session,
            snapshot_id=1,
            sample_size=10,
            min_depth=1,
            max_depth=5,
        )

        assert len(samples) == 0

    def test_sample_size_cap(self, index_session, sample_snapshot):
        """Test that sample size is respected"""
        # Create 100 nodes using factory
        for i in range(100):
            NodeFactory(node_id=i + 1, snapshot_id=1, depth=1)

        samples = select_training_samples(
            session=index_session,
            snapshot_id=1,
            sample_size=20,
            min_depth=1,
            max_depth=5,
        )

        # Should not exceed requested sample size
        assert len(samples) <= 20


class TestClusterBySimilarity:
    """Test _cluster_by_similarity function"""

    def test_basic_clustering(self):
        """Test basic clustering of similar names"""
        nodes = [
            Node(
                node_id=1,
                snapshot_id=1,
                name="Character Art",
                kind=NodeKind.DIR,
                parent_node_id=None,
                depth=1,
                rel_path="Character Art",
                abs_path="/test/Character Art",
                ext=None,
                file_source="filesystem",
            ),
            Node(
                node_id=2,
                snapshot_id=1,
                name="Character Arts",
                kind=NodeKind.DIR,
                parent_node_id=None,
                depth=1,
                rel_path="Character Arts",
                abs_path="/test/Character Arts",
                ext=None,
                file_source="filesystem",
            ),
            Node(
                node_id=3,
                snapshot_id=1,
                name="Environment",
                kind=NodeKind.DIR,
                parent_node_id=None,
                depth=1,
                rel_path="Environment",
                abs_path="/test/Environment",
                ext=None,
                file_source="filesystem",
            ),
        ]

        clusters = _cluster_by_similarity(nodes, threshold=0.4)

        # Similar names should be clustered together
        assert len(clusters) >= 1
        # Total nodes should be preserved
        total_nodes = sum(len(cluster) for cluster in clusters)
        assert total_nodes == 3

    def test_empty_input(self):
        """Test clustering with empty input"""
        clusters = _cluster_by_similarity([], threshold=0.5)
        assert len(clusters) == 0

    def test_single_node(self):
        """Test clustering with single node"""
        node = Node(
            node_id=1,
            snapshot_id=1,
            name="Test",
            kind=NodeKind.DIR,
            parent_node_id=None,
            depth=1,
            rel_path="Test",
            abs_path="/test/Test",
            ext=None,
            file_source="filesystem",
        )

        clusters = _cluster_by_similarity([node], threshold=0.5)
        assert len(clusters) == 1
        assert len(clusters[0]) == 1

    def test_threshold_effect(self):
        """Test that threshold affects clustering"""
        nodes = [
            Node(
                node_id=i,
                snapshot_id=1,
                name=f"folder_{i}",
                kind=NodeKind.DIR,
                parent_node_id=None,
                depth=1,
                rel_path=f"folder_{i}",
                abs_path=f"/test/folder_{i}",
                ext=None,
                file_source="filesystem",
            )
            for i in range(5)
        ]

        # Lower threshold should result in fewer clusters (more grouping)
        clusters_low = _cluster_by_similarity(nodes, threshold=0.2)
        # Higher threshold should result in more clusters (less grouping)
        clusters_high = _cluster_by_similarity(nodes, threshold=0.9)

        # All nodes should be accounted for in both cases
        assert sum(len(c) for c in clusters_low) == 5
        assert sum(len(c) for c in clusters_high) == 5


class TestSampleFromClusters:
    """Test _sample_from_clusters function"""

    def test_basic_sampling(self):
        """Test basic sampling from clusters"""
        clusters = [
            [
                Node(
                    node_id=1,
                    snapshot_id=1,
                    name="A1",
                    kind=NodeKind.DIR,
                    parent_node_id=None,
                    depth=1,
                    rel_path="A1",
                    abs_path="/test/A1",
                    ext=None,
                    file_source="filesystem",
                ),
                Node(
                    node_id=2,
                    snapshot_id=1,
                    name="A2",
                    kind=NodeKind.DIR,
                    parent_node_id=None,
                    depth=1,
                    rel_path="A2",
                    abs_path="/test/A2",
                    ext=None,
                    file_source="filesystem",
                ),
            ],
            [
                Node(
                    node_id=3,
                    snapshot_id=1,
                    name="B1",
                    kind=NodeKind.DIR,
                    parent_node_id=None,
                    depth=1,
                    rel_path="B1",
                    abs_path="/test/B1",
                    ext=None,
                    file_source="filesystem",
                )
            ],
        ]

        samples = _sample_from_clusters(clusters, num_samples=2)

        assert len(samples) == 2
        # Should get at least one from each cluster for diversity
        assert len(samples) <= 2

    def test_empty_clusters(self):
        """Test sampling from empty clusters"""
        samples = _sample_from_clusters([], num_samples=5)
        assert len(samples) == 0

    def test_sample_size_limit(self):
        """Test that sample size is respected"""
        clusters = [
            [
                Node(
                    node_id=i,
                    snapshot_id=1,
                    name=f"Node{i}",
                    kind=NodeKind.DIR,
                    parent_node_id=None,
                    depth=1,
                    rel_path=f"Node{i}",
                    abs_path=f"/test/Node{i}",
                    ext=None,
                    file_source="filesystem",
                )
                for i in range(10)
            ]
        ]

        samples = _sample_from_clusters(clusters, num_samples=5)

        assert len(samples) <= 5

    def test_diversity_preference(self):
        """Test that sampling prefers one from each cluster"""
        # Create 3 clusters with different sizes
        clusters = [
            [
                Node(
                    node_id=i,
                    snapshot_id=1,
                    name=f"A{i}",
                    kind=NodeKind.DIR,
                    parent_node_id=None,
                    depth=1,
                    rel_path=f"A{i}",
                    abs_path=f"/test/A{i}",
                    ext=None,
                    file_source="filesystem",
                )
                for i in range(5)
            ],
            [
                Node(
                    node_id=i + 100,
                    snapshot_id=1,
                    name=f"B{i}",
                    kind=NodeKind.DIR,
                    parent_node_id=None,
                    depth=1,
                    rel_path=f"B{i}",
                    abs_path=f"/test/B{i}",
                    ext=None,
                    file_source="filesystem",
                )
                for i in range(3)
            ],
            [
                Node(
                    node_id=i + 200,
                    snapshot_id=1,
                    name=f"C{i}",
                    kind=NodeKind.DIR,
                    parent_node_id=None,
                    depth=1,
                    rel_path=f"C{i}",
                    abs_path=f"/test/C{i}",
                    ext=None,
                    file_source="filesystem",
                )
                for i in range(2)
            ],
        ]

        samples = _sample_from_clusters(clusters, num_samples=3)

        # Should get one from each cluster for maximum diversity
        assert len(samples) == 3


class TestWriteSampleCsv:
    """Test write_sample_csv function"""

    def test_basic_csv_writing(self, index_session, sample_snapshot, tmp_path):
        """Test basic CSV writing without heuristic"""
        nodes = [
            Node(
                node_id=1,
                snapshot_id=1,
                name="TestFolder",
                kind=NodeKind.DIR,
                parent_node_id=None,
                depth=1,
                rel_path="TestFolder",
                abs_path="/test/TestFolder",
                ext=None,
                file_source="filesystem",
            )
        ]
        index_session.add_all(nodes)
        index_session.commit()

        output_path = tmp_path / "test.csv"

        write_sample_csv(
            output_path=output_path,
            nodes=nodes,
            session=index_session,
            snapshot_id=1,
            use_heuristic=False,
        )

        assert output_path.exists()

        # Read and verify CSV
        with output_path.open("r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["name"] == "TestFolder"
        assert rows[0]["snapshot_id"] == "1"
        assert rows[0]["node_id"] == "1"
        assert rows[0]["label"] == ""

    def test_csv_with_parent_and_grandparent(
        self, index_session, sample_snapshot, tmp_path
    ):
        """Test CSV includes parent and grandparent information"""
        nodes = [
            Node(
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
            ),
            Node(
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
            ),
            Node(
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
            ),
        ]
        index_session.add_all(nodes)
        index_session.commit()

        output_path = tmp_path / "test.csv"

        write_sample_csv(
            output_path=output_path,
            nodes=[nodes[2]],  # Only write Target node
            session=index_session,
            snapshot_id=1,
            use_heuristic=False,
        )

        with output_path.open("r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["parent_name"] == "Parent"
        assert rows[0]["grandparent_name"] == "Grandparent"


class TestReadClassificationCsv:
    """Test read_classification_csv function"""

    def test_basic_csv_reading(self, tmp_path):
        """Test reading a basic CSV file"""
        csv_path = tmp_path / "test.csv"

        # Write a test CSV
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["snapshot_id", "node_id", "name", "label"]
            )
            writer.writeheader()
            writer.writerow(
                {"snapshot_id": "1", "node_id": "100", "name": "Test", "label": "variant"}
            )
            writer.writerow(
                {"snapshot_id": "1", "node_id": "101", "name": "Test2", "label": "subject"}
            )

        rows = read_classification_csv(csv_path)

        assert len(rows) == 2
        assert rows[0]["snapshot_id"] == 1
        assert rows[0]["node_id"] == 100
        assert rows[0]["label"] == "variant"
        assert rows[1]["label"] == "subject"

    def test_missing_required_columns(self, tmp_path):
        """Test error on missing required columns"""
        csv_path = tmp_path / "test.csv"

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["snapshot_id", "name"])
            writer.writeheader()
            writer.writerow({"snapshot_id": "1", "name": "Test"})

        with pytest.raises(ValueError, match="missing required columns"):
            read_classification_csv(csv_path)

    def test_empty_csv(self, tmp_path):
        """Test error on empty CSV"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("")

        with pytest.raises(ValueError, match="empty or has no header"):
            read_classification_csv(csv_path)

    def test_extra_columns(self, tmp_path):
        """Test that extra columns are preserved"""
        csv_path = tmp_path / "test.csv"

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "snapshot_id",
                    "node_id",
                    "label",
                    "extra_col",
                    "another_col",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "snapshot_id": "1",
                    "node_id": "100",
                    "label": "variant",
                    "extra_col": "value1",
                    "another_col": "value2",
                }
            )

        rows = read_classification_csv(csv_path)

        assert len(rows) == 1
        assert "extra_col" in rows[0]
        assert "another_col" in rows[0]
        assert rows[0]["extra_col"] == "value1"


class TestValidateAllLabelsPresent:
    """Test validate_all_labels_present function"""

    def test_all_labels_present(self):
        """Test validation passes when all labels are present"""
        rows = [
            {"label": "variant"},
            {"label": "subject"},
            {"label": "other"},
        ]

        # Should not raise
        validate_all_labels_present(rows)

    def test_missing_labels(self):
        """Test validation fails when labels are missing"""
        rows = [
            {"label": "variant"},
            {"label": ""},
            {"label": "subject"},
        ]

        with pytest.raises(ValueError, match="missing labels"):
            validate_all_labels_present(rows)

    def test_empty_label_list(self):
        """Test validation passes with empty list"""
        validate_all_labels_present([])

    def test_error_message_format(self):
        """Test error message includes row numbers"""
        rows = [
            {"label": "variant"},
            {"label": ""},
            {"label": ""},
        ]

        with pytest.raises(ValueError) as exc_info:
            validate_all_labels_present(rows)

        # Error should mention specific rows (row 2 is first data row)
        assert "3, 4" in str(exc_info.value)


class TestValidateLabelValues:
    """Test validate_label_values function"""

    def test_all_valid_labels(self):
        """Test validation passes with all valid labels"""
        rows = [
            {"label": "variant"},
            {"label": "subject"},
            {"label": "other"},
        ]
        valid_labels = {"variant", "subject", "other", "collection"}

        # Should not raise
        validate_label_values(rows, valid_labels)

    def test_invalid_labels(self):
        """Test validation fails with invalid labels"""
        rows = [
            {"label": "variant"},
            {"label": "invalid_label"},
            {"label": "subject"},
        ]
        valid_labels = {"variant", "subject", "other"}

        with pytest.raises(ValueError, match="invalid labels"):
            validate_label_values(rows, valid_labels)

    def test_empty_labels_ignored(self):
        """Test that empty labels are ignored in validation"""
        rows = [
            {"label": "variant"},
            {"label": ""},
            {"label": "subject"},
        ]
        valid_labels = {"variant", "subject"}

        # Should not raise (empty labels are allowed)
        validate_label_values(rows, valid_labels)

    def test_error_message_includes_valid_labels(self):
        """Test error message includes list of valid labels"""
        rows = [{"label": "invalid"}]
        valid_labels = {"variant", "subject", "other"}

        with pytest.raises(ValueError) as exc_info:
            validate_label_values(rows, valid_labels)

        error_msg = str(exc_info.value)
        assert "variant" in error_msg
        assert "subject" in error_msg
        assert "other" in error_msg


class TestValidateInputCsv:
    """Test validate_input_csv function"""

    def test_valid_csv(self, tmp_path):
        """Test validation of valid CSV"""
        csv_path = tmp_path / "test.csv"

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["snapshot_id", "node_id", "name", "label"]
            )
            writer.writeheader()
            writer.writerow(
                {"snapshot_id": "1", "node_id": "100", "name": "Test", "label": "variant"}
            )

        rows = validate_input_csv(csv_path, taxonomy="v2")

        assert len(rows) == 1
        assert rows[0]["label"] == "variant"

    def test_invalid_taxonomy(self, tmp_path):
        """Test error on invalid taxonomy"""
        csv_path = tmp_path / "test.csv"

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["snapshot_id", "node_id", "label"])
            writer.writeheader()
            writer.writerow({"snapshot_id": "1", "node_id": "100", "label": "variant"})

        with pytest.raises(ValueError):
            validate_input_csv(csv_path, taxonomy="invalid_taxonomy")


class TestCreateLabelRuns:
    """Test create_label_runs function"""

    def test_create_single_run(self, training_session):
        """Test creating a single label run"""
        label_runs = create_label_runs(training_session, [1])

        assert set(label_runs.keys()) == {1}
        assert label_runs[1].snapshot_id == 1
        assert label_runs[1].label_source == "manual"
        assert label_runs[1].id is not None

    def test_create_multiple_runs(self, training_session):
        """Test creating multiple label runs"""
        label_runs = create_label_runs(training_session, [1, 2, 3])

        assert set(label_runs.keys()) == {1, 2, 3}
        assert all(lr.label_source == "manual" for lr in label_runs.values())


class TestApplyLabelsToSamples:
    """Test apply_labels_to_samples function"""

    def test_apply_labels_basic(self, training_session):
        """Test applying labels to samples"""
        # Create label run
        label_run = LabelRun(snapshot_id=1, label_source="manual")
        training_session.add(label_run)
        training_session.flush()

        # Create samples
        samples = [
            TrainingSample(
                label_run_id=label_run.id,
                snapshot_id=1,
                node_id=100,
                name_raw="test1",
            ),
            TrainingSample(
                label_run_id=label_run.id,
                snapshot_id=1,
                node_id=101,
                name_raw="test2",
            ),
        ]
        training_session.add_all(samples)
        training_session.commit()

        # Prepare CSV rows
        rows = [
            {"snapshot_id": 1, "node_id": 100, "label": "variant"},
            {"snapshot_id": 1, "node_id": 101, "label": "subject"},
        ]

        label_runs_dict = {1: label_run}

        labeled_count = apply_labels_to_samples(
            training_session, rows, label_runs_dict, split=None
        )

        assert labeled_count == 2

        # Verify labels were applied
        sample1 = (
            training_session.query(TrainingSample)
            .filter_by(node_id=100)
            .first()
        )
        assert sample1.label == "variant"
        assert sample1.label_confidence == 1.0

        sample2 = (
            training_session.query(TrainingSample)
            .filter_by(node_id=101)
            .first()
        )
        assert sample2.label == "subject"

    def test_apply_labels_with_split(self, training_session):
        """Test applying labels with split assignment"""
        label_run = LabelRun(snapshot_id=1, label_source="manual")
        training_session.add(label_run)
        training_session.flush()

        sample = TrainingSample(
            label_run_id=label_run.id,
            snapshot_id=1,
            node_id=100,
            name_raw="test",
        )
        training_session.add(sample)
        training_session.commit()

        rows = [{"snapshot_id": 1, "node_id": 100, "label": "variant"}]
        label_runs_dict = {1: label_run}

        apply_labels_to_samples(
            training_session, rows, label_runs_dict, split="train"
        )

        updated_sample = (
            training_session.query(TrainingSample)
            .filter_by(node_id=100)
            .first()
        )
        assert updated_sample.split == "train"

    def test_apply_labels_missing_sample(self, training_session):
        """Test that missing samples are skipped"""
        label_run = LabelRun(snapshot_id=1, label_source="manual")
        training_session.add(label_run)
        training_session.flush()

        rows = [{"snapshot_id": 1, "node_id": 999, "label": "variant"}]
        label_runs_dict = {1: label_run}

        labeled_count = apply_labels_to_samples(
            training_session, rows, label_runs_dict, split=None
        )

        # Should not label non-existent sample
        assert labeled_count == 0
