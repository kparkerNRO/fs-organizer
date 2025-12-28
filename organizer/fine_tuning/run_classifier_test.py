"""Tests for run_classifier.py"""

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from storage.training_models import (
    ModelRun,
    SamplePrediction,
    TrainingBase,
    TrainingSample,
)

from fine_tuning.run_classifier import (
    create_model_run,
    evaluate_predictions,
    load_samples,
    save_predictions_to_db,
)
from fine_tuning.taxonomy import (
    LABELS_LEGACY,
    LABELS_V1,
    LABELS_V2,
)


@pytest.fixture
def db_session():
    """Create in-memory test database"""
    engine = create_engine("sqlite:///:memory:")
    TrainingBase.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


@pytest.fixture
def sample_training_samples(db_session):
    """Create sample training data"""
    samples = [
        TrainingSample(
            snapshot_id=1,
            node_id=1,
            name_raw="John Doe",
            name_norm="john doe",
            kind="directory",
            file_source="filesystem",
            depth=1,
            text="gp: | p: | t:john doe | depth:1 | sibs: | children: | exts: | flags:collab=0 childMedia=0",
            label="primary_author",
            split="train",
        ),
        TrainingSample(
            snapshot_id=1,
            node_id=2,
            name_raw="Character Art",
            name_norm="character art",
            kind="directory",
            file_source="filesystem",
            depth=1,
            text="gp: | p: | t:character art | depth:1 | sibs: | children: | exts:png jpg | flags:collab=0 childMedia=1",
            label="subject",
            split="train",
        ),
        TrainingSample(
            snapshot_id=1,
            node_id=3,
            name_raw="High Resolution",
            name_norm="high resolution",
            kind="directory",
            file_source="filesystem",
            depth=2,
            text="gp: | p:Character Art | t:high resolution | depth:2 | sibs: | children: | exts:png | flags:collab=0 childMedia=0",
            label="variant",
            split="test",
        ),
        TrainingSample(
            snapshot_id=1,
            node_id=4,
            name_raw="Unlabeled Folder",
            name_norm="unlabeled folder",
            kind="directory",
            file_source="filesystem",
            depth=1,
            text="gp: | p: | t:unlabeled folder | depth:1 | sibs: | children: | exts: | flags:collab=0 childMedia=0",
            label=None,
            split="train",
        ),
    ]

    db_session.add_all(samples)
    db_session.commit()

    return samples


class TestLoadSamples:
    """Test load_samples function"""

    def test_load_all_samples(self, db_session, sample_training_samples):
        """Load all samples without filters"""
        samples = load_samples(db_session)
        assert len(samples) == 4

    def test_load_by_split(self, db_session, sample_training_samples):
        """Load samples filtered by split"""
        train_samples = load_samples(db_session, split="train")
        assert len(train_samples) == 3
        assert all(s.split == "train" for s in train_samples)

        test_samples = load_samples(db_session, split="test")
        assert len(test_samples) == 1
        assert test_samples[0].split == "test"

    def test_load_labeled_only(self, db_session, sample_training_samples):
        """Load only labeled samples"""
        labeled = load_samples(db_session, labeled_only=True)
        assert len(labeled) == 3
        assert all(s.label is not None and s.label != "" for s in labeled)

    def test_load_split_and_labeled(self, db_session, sample_training_samples):
        """Load with both filters"""
        samples = load_samples(db_session, split="train", labeled_only=True)
        assert len(samples) == 2
        assert all(s.split == "train" and s.label is not None for s in samples)


class TestEvaluatePredictions:
    """Test evaluate_predictions function"""

    @pytest.mark.parametrize(
        "labels,expected_accuracy",
        [
            (LABELS_LEGACY, 0.75),
            (LABELS_V1, 0.75),
            (LABELS_V2, 0.75),
        ],
    )
    def test_evaluate_predictions_basic(self, labels, expected_accuracy):
        """Test basic evaluation metrics"""
        y_true = ["primary_author", "subject", "variant", "other"]
        y_pred = ["primary_author", "subject", "variant", "subject"]  # last one wrong

        metrics = evaluate_predictions(y_true, y_pred, labels, verbose=False)

        assert metrics["accuracy"] == expected_accuracy
        assert metrics["num_samples"] == 4
        assert 0 <= metrics["macro_f1"] <= 1
        assert 0 <= metrics["weighted_f1"] <= 1

    def test_perfect_predictions(self):
        """Test with perfect predictions"""
        y_true = ["primary_author", "subject", "variant"]
        y_pred = ["primary_author", "subject", "variant"]

        # Only use labels that are actually present in the data
        labels_used = {"primary_author", "subject", "variant"}

        metrics = evaluate_predictions(y_true, y_pred, labels_used, verbose=False)

        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["weighted_f1"] == 1.0

    def test_all_wrong_predictions(self):
        """Test with all wrong predictions"""
        y_true = ["primary_author", "subject", "variant"]
        y_pred = ["other", "other", "other"]

        metrics = evaluate_predictions(y_true, y_pred, LABELS_LEGACY, verbose=False)

        assert metrics["accuracy"] == 0.0


class TestSavePredictionsToDb:
    """Test save_predictions_to_db function"""

    def test_save_predictions(self, db_session, sample_training_samples):
        """Test saving predictions to database"""
        # Create a model run first
        run = create_model_run(
            db_session,
            model_path="test_model",
            taxonomy="legacy",
            use_baseline=False,
            config={"test": True},
        )

        # Create predictions
        samples = sample_training_samples[:2]
        predictions = ["primary_author", "variant"]  # second is wrong
        confidences = [0.95, 0.82]
        probabilities = [
            {"primary_author": 0.95, "subject": 0.03, "variant": 0.02},
            {"primary_author": 0.10, "subject": 0.08, "variant": 0.82},
        ]

        # Save predictions
        num_saved = save_predictions_to_db(
            db_session,
            samples,
            predictions,
            confidences,
            probabilities,
            run_id=run.run_id,
            prediction_type="test",
        )

        assert num_saved == 2

        # Verify saved predictions
        saved = db_session.query(SamplePrediction).all()
        assert len(saved) == 2

        # Check first prediction (correct)
        pred1 = saved[0]
        assert pred1.predicted_label == "primary_author"
        assert pred1.true_label == "primary_author"
        assert pred1.is_correct is True
        assert pred1.confidence == 0.95
        assert pred1.prediction_type == "test"

        # Check second prediction (incorrect)
        pred2 = saved[1]
        assert pred2.predicted_label == "variant"
        assert pred2.true_label == "subject"
        assert pred2.is_correct is False
        assert pred2.confidence == 0.82

    def test_save_predictions_without_labels(self, db_session):
        """Test saving predictions for unlabeled samples"""
        # Create unlabeled sample
        sample = TrainingSample(
            snapshot_id=1,
            node_id=100,
            name_raw="Unlabeled",
            name_norm="unlabeled",
            kind="directory",
            file_source="filesystem",
            depth=1,
            text="test",
            label=None,
        )
        db_session.add(sample)
        db_session.commit()

        # Create model run
        run = create_model_run(
            db_session,
            model_path="test_model",
            taxonomy="legacy",
            use_baseline=False,
            config={},
        )

        # Save prediction
        num_saved = save_predictions_to_db(
            db_session,
            [sample],
            ["subject"],
            [0.75],
            [{"subject": 0.75}],
            run_id=run.run_id,
        )

        assert num_saved == 1

        pred = db_session.query(SamplePrediction).first()
        assert pred.predicted_label == "subject"
        assert pred.true_label is None
        assert pred.is_correct is None


class TestCreateModelRun:
    """Test create_model_run function"""

    def test_create_baseline_run(self, db_session):
        """Test creating baseline model run"""
        run = create_model_run(
            db_session,
            model_path=None,
            taxonomy="v2",
            use_baseline=True,
            config={"batch_size": 32},
        )

        assert run.run_id is not None
        assert run.status == "running"
        assert run.run_type == "baseline"
        assert run.taxonomy == "v2"
        assert run.model_type == "setfit"
        assert "baseline-v2" in run.model_version
        assert "setfit-baseline-v2" in run.base_model_id

        # Check config stored
        stored_config = json.loads(run.hyperparameters_json)
        assert stored_config["batch_size"] == 32

    def test_create_finetuned_run(self, db_session):
        """Test creating fine-tuned model run"""
        run = create_model_run(
            db_session,
            model_path="/path/to/model",
            taxonomy="v1",
            use_baseline=False,
            config={"learning_rate": 0.001},
            run_type="training",
        )

        assert run.run_type == "training"
        assert run.taxonomy == "v1"
        assert run.model_version == "/path/to/model"
        assert "setfit-training-v1" in run.base_model_id

        stored_config = json.loads(run.hyperparameters_json)
        assert stored_config["learning_rate"] == 0.001

    def test_create_run_with_metadata(self, db_session):
        """Test creating run with additional metadata"""
        run = create_model_run(
            db_session,
            model_path="test_model",
            taxonomy="legacy",
            use_baseline=False,
            config={},
            run_type="evaluation",
            training_data_source="snapshot_123",
        )

        assert run.run_type == "evaluation"
        assert run.training_data_source == "snapshot_123"
        assert run.started_at is not None


class TestLabelTaxonomies:
    """Test label taxonomy definitions"""

    def test_v1_labels(self):
        """Verify V1 label taxonomy"""
        assert "person_or_group" in LABELS_V1
        assert "content" in LABELS_V1
        assert "media_bucket" in LABELS_V1
        assert "descriptor" in LABELS_V1
        assert "other" in LABELS_V1
        assert "unknown" in LABELS_V1
        assert len(LABELS_V1) == 6

    def test_v2_labels(self):
        """Verify V2 label taxonomy"""
        assert "creator_or_studio" in LABELS_V2
        assert "content_subject" in LABELS_V2
        assert "descriptor" in LABELS_V2
        assert "asset_type" in LABELS_V2
        assert "other" in LABELS_V2
        assert "unknown" in LABELS_V2
        assert len(LABELS_V2) == 6

    def test_legacy_labels(self):
        """Verify legacy label taxonomy"""
        assert "primary_author" in LABELS_LEGACY
        assert "secondary_author" in LABELS_LEGACY
        assert "collection" in LABELS_LEGACY
        assert "subject" in LABELS_LEGACY
        assert "media_format" in LABELS_LEGACY
        assert "media_type" in LABELS_LEGACY
        assert "variant" in LABELS_LEGACY
        assert "other" in LABELS_LEGACY
        assert len(LABELS_LEGACY) == 8
