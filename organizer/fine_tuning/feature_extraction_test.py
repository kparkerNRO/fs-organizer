"""Tests for feature_extraction.py"""

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from storage.index_models import IndexBase, Node, Snapshot
from storage.manager import NodeKind
from fine_tuning.training_models import TrainingBase, TrainingSample, LabelRun
from utils.config import Config
from utils.text_processing import has_matching_token

from fine_tuning.feature_extraction import (
    extract_features,
)


@pytest.fixture
def index_session():
    """Create in-memory test database for index data"""
    engine = create_engine("sqlite:///:memory:")
    IndexBase.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


@pytest.fixture
def training_session():
    """Create in-memory test database for training data"""
    engine = create_engine("sqlite:///:memory:")
    TrainingBase.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


@pytest.fixture
def test_config():
    """Create test configuration with cue markers"""
    # Create minimal Config for testing
    config = Config(
        creators={},
        creator_removes={},
        creator_strings=set(),
        file_name_exceptions={},
        replace_exceptions={},
        clean_exceptions=set(),
        should_ignore=set(),
        grouping_exceptions=(),
        variants={},
        known_variant_tokens=set(),
        variant_type_by_string={},
        variant_grouping_by_string={},
        variant_types={"high", "low", "res", "resolution", "quality"},
        media_types={"video", "audio", "image", "3d"},
        format_types={"png", "jpg", "mp4", "wav", "pdf"},
        relational_cache={},
    )
    return config


@pytest.fixture
def label_run(training_session):
    """Create a test label run"""
    label_run = LabelRun(snapshot_id=1, label_source="test")
    training_session.add(label_run)
    training_session.flush()
    return label_run


@pytest.fixture
def sample_nodes(index_session):
    """Create sample node hierarchy"""
    # Create snapshot first (required for foreign key)
    snapshot = Snapshot(
        snapshot_id=1,
        created_at="2024-01-01T00:00:00",
        root_path="/test",
        root_abs_path="/test",
    )
    index_session.add(snapshot)
    index_session.flush()

    # Root
    root = Node(
        node_id=1,
        snapshot_id=1,
        name="Artist Name",
        kind=NodeKind.DIR,
        parent_node_id=None,
        depth=0,
        rel_path="Artist Name",
        abs_path="/test/Artist Name",
        ext=None,
        file_source="filesystem",
    )

    # Level 1 - Collections
    collection1 = Node(
        node_id=2,
        snapshot_id=1,
        name="Character Art",
        kind=NodeKind.DIR,
        parent_node_id=1,
        depth=1,
        rel_path="Artist Name/Character Art",
        abs_path="/test/Artist Name/Character Art",
        ext=None,
        file_source="filesystem",
    )

    collection2 = Node(
        node_id=3,
        snapshot_id=1,
        name="Environment Scenes",
        kind=NodeKind.DIR,
        parent_node_id=1,
        depth=1,
        rel_path="Artist Name/Environment Scenes",
        abs_path="/test/Artist Name/Environment Scenes",
        ext=None,
        file_source="filesystem",
    )

    # Level 2 - Variant folder
    high_res = Node(
        node_id=4,
        snapshot_id=1,
        name="High Resolution",
        kind=NodeKind.DIR,
        parent_node_id=2,
        depth=2,
        rel_path="Artist Name/Character Art/High Resolution",
        abs_path="/test/Artist Name/Character Art/High Resolution",
        ext=None,
        file_source="filesystem",
    )

    low_res = Node(
        node_id=5,
        snapshot_id=1,
        name="Low Res",
        kind=NodeKind.DIR,
        parent_node_id=2,
        depth=2,
        rel_path="Artist Name/Character Art/Low Res",
        abs_path="/test/Artist Name/Character Art/Low Res",
        ext=None,
        file_source="filesystem",
    )

    # Level 3 - Files
    file1 = Node(
        node_id=6,
        snapshot_id=1,
        name="warrior.png",
        kind=NodeKind.FILE,
        parent_node_id=4,
        depth=3,
        rel_path="Artist Name/Character Art/High Resolution/warrior.png",
        abs_path="/test/Artist Name/Character Art/High Resolution/warrior.png",
        ext="png",
        file_source="filesystem",
    )

    file2 = Node(
        node_id=7,
        snapshot_id=1,
        name="mage.jpg",
        kind=NodeKind.FILE,
        parent_node_id=4,
        depth=3,
        rel_path="Artist Name/Character Art/High Resolution/mage.jpg",
        abs_path="/test/Artist Name/Character Art/High Resolution/mage.jpg",
        ext="jpg",
        file_source="filesystem",
    )

    file3 = Node(
        node_id=8,
        snapshot_id=1,
        name="scene01.png",
        kind=NodeKind.FILE,
        parent_node_id=3,
        depth=2,
        rel_path="Artist Name/Environment Scenes/scene01.png",
        abs_path="/test/Artist Name/Environment Scenes/scene01.png",
        ext="png",
        file_source="filesystem",
    )

    # ZIP file treated as container
    zip_node = Node(
        node_id=9,
        snapshot_id=1,
        name="assets.zip",
        kind=NodeKind.FILE,
        parent_node_id=1,
        depth=1,
        rel_path="Artist Name/assets.zip",
        abs_path="/test/Artist Name/assets.zip",
        ext="zip",
        file_source="zip_file",
    )

    # Content inside ZIP
    zip_content = Node(
        node_id=10,
        snapshot_id=1,
        name="texture.png",
        kind=NodeKind.FILE,
        parent_node_id=9,
        depth=2,
        rel_path="Artist Name/assets.zip/texture.png",
        abs_path="/test/Artist Name/assets.zip/texture.png",
        ext="png",
        file_source="zip_file",
    )

    nodes = [
        root,
        collection1,
        collection2,
        high_res,
        low_res,
        file1,
        file2,
        file3,
        zip_node,
        zip_content,
    ]

    index_session.add_all(nodes)
    index_session.commit()

    return nodes


class TestHasAnyToken:
    """Test has_any_token function"""

    def test_has_matching_token(self):
        """Test when token matches cue set"""
        token_list = ["character", "art", "high", "resolution"]
        cue_set = {"high", "low", "medium"}

        assert has_matching_token(token_list, cue_set) is True

    def test_no_matching_token(self):
        """Test when no token matches"""
        token_list = ["character", "art", "texture"]
        cue_set = {"video", "audio", "3d"}

        assert has_matching_token(token_list, cue_set) is False

    def test_empty_token_list(self):
        """Test with empty token list"""
        assert has_matching_token([], {"video", "audio"}) is False

    def test_empty_cue_set(self):
        """Test with empty cue set"""
        assert has_matching_token(["test"], set()) is False

    def test_multiple_matches(self):
        """Test with multiple matching tokens"""
        token_list = ["video", "audio", "content"]
        cue_set = {"video", "audio", "image"}

        assert has_matching_token(token_list, cue_set) is True


class TestExtractFeatures:
    """Test extract_features function"""

    def test_basic_feature_extraction(
        self, index_session, training_session, sample_nodes, test_config, label_run
    ):
        """Test basic feature extraction from nodes"""
        num_created = extract_features(
            index_session,
            training_session,
            snapshot_id=1,
            config=test_config,
            label_run=label_run,
        )

        # Should create one sample per node, but only directories and ZIP files
        # From sample_nodes: root, collection1, collection2, high_res, low_res, zip_node = 6 containers
        # (regular files are skipped in feature extraction - only dirs and zip files are processed)
        assert num_created == 6

        # Verify samples were created
        samples = training_session.query(TrainingSample).all()
        assert len(samples) == 6

    def test_parent_child_relationships(
        self, index_session, training_session, sample_nodes, test_config, label_run
    ):
        """Test that parent-child relationships are captured"""
        extract_features(
            index_session,
            training_session,
            snapshot_id=1,
            config=test_config,
            label_run=label_run,
        )

        # Find the "High Resolution" folder sample
        high_res_sample = (
            training_session.query(TrainingSample)
            .filter(TrainingSample.name_raw == "High Resolution")
            .first()
        )

        assert high_res_sample is not None
        assert high_res_sample.parent_name_norm == "Character Art"
        assert high_res_sample.grandparent_name_norm == "Artist Name"
        assert high_res_sample.depth == 2

    def test_child_names_captured(
        self, index_session, training_session, sample_nodes, test_config, label_run
    ):
        """Test that child names are captured"""
        extract_features(
            index_session,
            training_session,
            snapshot_id=1,
            config=test_config,
            label_run=label_run,
        )

        # Find the "High Resolution" folder which has two children
        high_res_sample = (
            training_session.query(TrainingSample)
            .filter(TrainingSample.name_raw == "High Resolution")
            .first()
        )

        child_names = json.loads(high_res_sample.child_names_topk_json)
        assert len(child_names) == 2
        # Child names include extensions and are normalized
        assert any("mage" in name for name in child_names)
        assert any("warrior" in name for name in child_names)

    def test_sibling_names_captured(
        self, index_session, training_session, sample_nodes, test_config, label_run
    ):
        """Test that sibling names are captured"""
        extract_features(
            index_session,
            training_session,
            snapshot_id=1,
            config=test_config,
            label_run=label_run,
        )

        # "High Resolution" has sibling "Low Res"
        high_res_sample = (
            training_session.query(TrainingSample)
            .filter(TrainingSample.name_raw == "High Resolution")
            .first()
        )

        sibling_names = json.loads(high_res_sample.sibling_names_topk_json)
        # Sibling names are normalized via _processed_name
        assert any(
            "low" in name.lower() and "res" in name.lower() for name in sibling_names
        )

    def test_descendant_extensions(
        self, index_session, training_session, sample_nodes, test_config, label_run
    ):
        """Test that descendant file extensions are captured"""
        extract_features(
            index_session,
            training_session,
            snapshot_id=1,
            config=test_config,
            label_run=label_run,
        )

        # "Character Art" folder contains descendant files with .png and .jpg
        char_art_sample = (
            training_session.query(TrainingSample)
            .filter(TrainingSample.name_raw == "Character Art")
            .first()
        )

        exts = json.loads(char_art_sample.descendant_file_exts_topk_json)
        assert "png" in exts or "jpg" in exts

    def test_cue_detection(
        self, index_session, training_session, sample_nodes, test_config, label_run
    ):
        """Test that cue markers are detected"""
        extract_features(
            index_session,
            training_session,
            snapshot_id=1,
            config=test_config,
            label_run=label_run,
        )

        # "High Resolution" should have variant hint from children
        high_res_sample = (
            training_session.query(TrainingSample)
            .filter(TrainingSample.name_raw == "High Resolution")
            .first()
        )

        # Should detect "resolution" as variant hint
        assert high_res_sample.sibling_has_variant_hint is True

    def test_text_field_format(
        self, index_session, training_session, sample_nodes, test_config, label_run
    ):
        """Test that text field is properly formatted"""
        extract_features(
            index_session,
            training_session,
            snapshot_id=1,
            config=test_config,
            label_run=label_run,
        )

        sample = (
            training_session.query(TrainingSample)
            .filter(TrainingSample.name_raw == "High Resolution")
            .first()
        )

        # Check text field contains expected components
        assert "gp:" in sample.text
        assert "p:" in sample.text
        assert "t:" in sample.text
        assert "depth:" in sample.text
        assert "sibs:" in sample.text
        assert "children:" in sample.text
        assert "exts:" in sample.text
        assert "flags:" in sample.text

    def test_zip_file_handling(
        self, index_session, training_session, sample_nodes, test_config, label_run
    ):
        """Test that ZIP files are treated as containers"""
        extract_features(
            index_session,
            training_session,
            snapshot_id=1,
            config=test_config,
            label_run=label_run,
        )

        # Find the ZIP file sample
        zip_sample = (
            training_session.query(TrainingSample)
            .filter(TrainingSample.name_raw == "assets.zip")
            .first()
        )

        assert zip_sample is not None
        assert zip_sample.file_source == "zip_file"

        # ZIP file should have the internal file as a child
        child_names = json.loads(zip_sample.child_names_topk_json)
        assert any("texture" in name for name in child_names)

    def test_batch_processing(
        self, index_session, training_session, sample_nodes, test_config, label_run
    ):
        """Test batch processing with small batch size"""
        num_created = extract_features(
            index_session,
            training_session,
            snapshot_id=1,
            config=test_config,
            label_run=label_run,
            batch_size=3,  # Small batch size to test batching logic
        )

        # Only directories and ZIP files are processed
        assert num_created == 6

        # All samples should still be created
        samples = training_session.query(TrainingSample).all()
        assert len(samples) == 6

    def test_child_cap(self, index_session, training_session, test_config, label_run):
        """Test that child_cap limits number of children stored"""
        # Create snapshot first
        snapshot = Snapshot(
            snapshot_id=1,
            created_at="2024-01-01T00:00:00",
            root_path="/test",
            root_abs_path="/test",
        )
        index_session.add(snapshot)
        index_session.flush()

        # Create a node with many children
        parent = Node(
            node_id=100,
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

        # Add 25 children
        for i in range(25):
            child = Node(
                node_id=200 + i,
                snapshot_id=1,
                name=f"child_{i:02d}",
                kind=NodeKind.FILE,
                parent_node_id=100,
                depth=1,
                rel_path=f"Parent/child_{i:02d}",
                abs_path=f"/test/Parent/child_{i:02d}",
                ext="txt",
                file_source="filesystem",
            )
            index_session.add(child)

        index_session.commit()

        extract_features(
            index_session,
            training_session,
            snapshot_id=1,
            config=test_config,
            label_run=label_run,
            child_cap=10,
        )

        parent_sample = (
            training_session.query(TrainingSample)
            .filter(TrainingSample.name_raw == "Parent")
            .first()
        )

        child_names = json.loads(parent_sample.child_names_topk_json)
        assert len(child_names) <= 10

    def test_empty_snapshot(
        self, index_session, training_session, test_config, label_run
    ):
        """Test extraction with non-existent snapshot"""
        # Create a label_run for snapshot 999
        label_run_999 = LabelRun(snapshot_id=999, label_source="test")
        training_session.add(label_run_999)
        training_session.flush()

        num_created = extract_features(
            index_session,
            training_session,
            snapshot_id=999,  # Non-existent
            config=test_config,
            label_run=label_run_999,
        )

        assert num_created == 0

        samples = training_session.query(TrainingSample).all()
        assert len(samples) == 0
