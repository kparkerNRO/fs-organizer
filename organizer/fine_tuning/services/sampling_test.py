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

        # Should only include nodes at depth 2 and 5 (not depth 0)
        assert len(samples) == 2
        depths = {s.depth for s in samples}
        assert depths == {2, 5}

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

        # Should return exactly the requested sample size
        assert len(samples) == 20


class TestClusterBySimilarity:
    """Test _cluster_by_similarity function"""

    def test_basic_clustering(self):
        """Test basic clustering of similar names"""
        nodes = [
            NodeFactory.build(node_id=1, name="Character Art"),
            NodeFactory.build(node_id=2, name="Character Arts"),
            NodeFactory.build(node_id=3, name="Environment"),
        ]

        clusters = _cluster_by_similarity(nodes, threshold=0.4)

        # "Character Art" and "Character Arts" are similar → 1 cluster
        # "Environment" is dissimilar → separate cluster
        assert len(clusters) == 2

        # Verify cluster contents
        cluster_names = [sorted([node.name for node in cluster]) for cluster in clusters]
        cluster_names.sort()  # Sort for deterministic comparison

        assert cluster_names == [
            ["Character Art", "Character Arts"],
            ["Environment"],
        ]

    def test_empty_input(self):
        """Test clustering with empty input"""
        clusters = _cluster_by_similarity([], threshold=0.5)
        assert len(clusters) == 0

    def test_single_node(self):
        """Test clustering with single node"""
        node = NodeFactory.build(node_id=1, name="Test")

        clusters = _cluster_by_similarity([node], threshold=0.5)
        assert len(clusters) == 1
        assert len(clusters[0]) == 1

    def test_threshold_effect(self):
        """Test that threshold affects clustering behavior"""
        # Create nodes with varying similarity:
        # - "Image Res" and "Image Resolution" are similar (high similarity)
        # - "Folder ABC" and "Folder XYZ" are moderately similar (same prefix)
        # - "Random Data" is dissimilar to all
        nodes = [
            NodeFactory.build(node_id=1, name="Image Res"),
            NodeFactory.build(node_id=2, name="Image Resolution"),
            NodeFactory.build(node_id=3, name="Folder ABC"),
            NodeFactory.build(node_id=4, name="Folder XYZ"),
            NodeFactory.build(node_id=5, name="Random Data"),
        ]

        # Low threshold (0.2): More permissive, groups more items together
        clusters_low = _cluster_by_similarity(nodes, threshold=0.2)
        # High threshold (0.9): Strict, only very similar items cluster
        clusters_high = _cluster_by_similarity(nodes, threshold=0.9)

        # Verify low threshold groups similar items
        assert len(clusters_low) == 3
        low_cluster_names = [sorted([node.name for node in cluster]) for cluster in clusters_low]
        low_cluster_names.sort()
        assert low_cluster_names == [
            ["Folder ABC", "Folder XYZ"],
            ["Image Res", "Image Resolution"],
            ["Random Data"],
        ]

        # Verify high threshold separates all items
        assert len(clusters_high) == 5
        high_cluster_names = [sorted([node.name for node in cluster]) for cluster in clusters_high]
        high_cluster_names.sort()
        assert high_cluster_names == [
            ["Folder ABC"],
            ["Folder XYZ"],
            ["Image Res"],
            ["Image Resolution"],
            ["Random Data"],
        ]


class TestSampleFromClusters:
    """Test _sample_from_clusters function"""

    def test_basic_sampling(self):
        """Test basic sampling from clusters"""
        clusters = [
            [
                NodeFactory.build(node_id=1, name="A1"),
                NodeFactory.build(node_id=2, name="A2"),
            ],
            [NodeFactory.build(node_id=3, name="B1")],
        ]

        samples = _sample_from_clusters(clusters, num_samples=2)

        assert len(samples) == 2

    def test_empty_clusters(self):
        """Test sampling from empty clusters"""
        samples = _sample_from_clusters([], num_samples=5)
        assert len(samples) == 0

    def test_sample_size_limit(self):
        """Test that sample size is exactly respected"""
        clusters = [[NodeFactory.build(node_id=i, name=f"Node{i}") for i in range(10)]]

        samples = _sample_from_clusters(clusters, num_samples=5)

        # Should return exactly the requested number of samples
        assert len(samples) == 5

    def test_diversity_preference(self):
        """Test that sampling prefers one from each cluster"""
        # Create 3 clusters with different sizes
        clusters = [
            [NodeFactory.build(node_id=i, name=f"A{i}") for i in range(5)],
            [NodeFactory.build(node_id=i + 100, name=f"B{i}") for i in range(3)],
            [NodeFactory.build(node_id=i + 200, name=f"C{i}") for i in range(2)],
        ]

        samples = _sample_from_clusters(clusters, num_samples=3)

        # Should get one from each cluster for maximum diversity
        assert len(samples) == 3

        # Verify samples come from different clusters based on node_id ranges
        sample_ids = {s.node_id for s in samples}
        has_from_cluster_a = any(id < 100 for id in sample_ids)
        has_from_cluster_b = any(100 <= id < 200 for id in sample_ids)
        has_from_cluster_c = any(200 <= id < 300 for id in sample_ids)

        assert has_from_cluster_a and has_from_cluster_b and has_from_cluster_c


class TestWriteSampleCsv:
    """Test write_sample_csv function"""

    def test_basic_csv_writing(self, index_session, sample_snapshot, tmp_path):
        """Test basic CSV writing without heuristic"""
        node = NodeFactory(node_id=1, snapshot_id=1, name="TestFolder", depth=1)

        output_path = tmp_path / "test.csv"

        write_sample_csv(
            output_path=output_path,
            nodes=[node],
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
        grandparent = NodeFactory(
            node_id=1, snapshot_id=1, name="Grandparent", parent_node_id=None, depth=0
        )
        parent = NodeFactory(
            node_id=2, snapshot_id=1, name="Parent", parent_node_id=1, depth=1
        )
        target = NodeFactory(
            node_id=3, snapshot_id=1, name="Target", parent_node_id=2, depth=2
        )

        output_path = tmp_path / "test.csv"

        write_sample_csv(
            output_path=output_path,
            nodes=[target],
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

    @pytest.mark.parametrize(
        "rows,valid_labels,should_raise,error_match",
        [
            # All valid labels - should pass
            (
                [{"label": "variant"}, {"label": "subject"}, {"label": "other"}],
                {"variant", "subject", "other", "collection"},
                False,
                None,
            ),
            # Empty labels ignored - should pass
            (
                [{"label": "variant"}, {"label": ""}, {"label": "subject"}],
                {"variant", "subject"},
                False,
                None,
            ),
            # Invalid label - should fail
            (
                [{"label": "variant"}, {"label": "invalid_label"}, {"label": "subject"}],
                {"variant", "subject", "other"},
                True,
                "invalid labels",
            ),
        ],
    )
    def test_label_validation(self, rows, valid_labels, should_raise, error_match):
        """Test label validation with various inputs"""
        if should_raise:
            with pytest.raises(ValueError, match=error_match):
                validate_label_values(rows, valid_labels)
        else:
            # Should not raise
            validate_label_values(rows, valid_labels)

    def test_error_message_includes_valid_labels(self):
        """Test error message includes list of valid labels"""
        rows = [{"label": "invalid"}]
        valid_labels = {"variant", "subject", "other"}

        with pytest.raises(ValueError) as exc_info:
            validate_label_values(rows, valid_labels)

        error_msg = str(exc_info.value)
        # Verify all valid labels are mentioned in error
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

    def test_apply_labels_basic(self, training_session, label_run):
        """Test applying labels to samples"""
        # Create samples
        TrainingSampleFactory(
            label_run_id=label_run.id, snapshot_id=1, node_id=100, name_raw="test1"
        )
        TrainingSampleFactory(
            label_run_id=label_run.id, snapshot_id=1, node_id=101, name_raw="test2"
        )

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
        from storage.training_models import TrainingSample

        sample1 = training_session.query(TrainingSample).filter_by(node_id=100).first()
        assert sample1.label == "variant"
        assert sample1.label_confidence == 1.0

        sample2 = training_session.query(TrainingSample).filter_by(node_id=101).first()
        assert sample2.label == "subject"

    def test_apply_labels_with_split(self, training_session, label_run):
        """Test applying labels with split assignment"""
        TrainingSampleFactory(
            label_run_id=label_run.id, snapshot_id=1, node_id=100, name_raw="test"
        )

        rows = [{"snapshot_id": 1, "node_id": 100, "label": "variant"}]
        label_runs_dict = {1: label_run}

        apply_labels_to_samples(
            training_session, rows, label_runs_dict, split="train"
        )

        from storage.training_models import TrainingSample

        updated_sample = (
            training_session.query(TrainingSample).filter_by(node_id=100).first()
        )
        assert updated_sample.split == "train"

    def test_apply_labels_missing_sample(self, training_session, label_run):
        """Test that missing samples are skipped"""
        rows = [{"snapshot_id": 1, "node_id": 999, "label": "variant"}]
        label_runs_dict = {1: label_run}

        labeled_count = apply_labels_to_samples(
            training_session, rows, label_runs_dict, split=None
        )

        # Should not label non-existent sample
        assert labeled_count == 0
