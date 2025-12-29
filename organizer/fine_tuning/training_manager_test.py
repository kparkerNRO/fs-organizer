"""Tests for training_manager.py"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from fine_tuning.training_models import (
    LabelRun,
    SamplePrediction,
    TrainingBase,
    TrainingSample,
)
from fine_tuning.training_manager import (
    create_model_run,
    get_newest_label_run_id,
    load_samples,
    save_predictions_to_db,
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
    # Create a label run first
    label_run = LabelRun(snapshot_id=1, label_source="test")
    db_session.add(label_run)
    db_session.flush()

    samples = [
        TrainingSample(
            snapshot_id=1,
            node_id=1,
            label_run_id=label_run.id,
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
            label_run_id=label_run.id,
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
            label_run_id=label_run.id,
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
            label_run_id=label_run.id,
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

    def test_load_by_label_run_id(self, db_session, sample_training_samples):
        """Load samples filtered by label_run_id"""
        label_run2 = LabelRun(snapshot_id=2, label_source="test2")
        db_session.add(label_run2)
        db_session.flush()

        sample2 = TrainingSample(
            snapshot_id=2,
            node_id=5,
            label_run_id=label_run2.id,
            name_raw="Test",
            name_norm="test",
            kind="directory",
            file_source="filesystem",
            depth=1,
            text="test",
            label="other",
            split="train",
        )
        db_session.add(sample2)
        db_session.commit()

        label_run1_id = sample_training_samples[0].label_run_id
        samples1 = load_samples(db_session, label_run_id=label_run1_id)
        assert len(samples1) == 4
        assert all(s.label_run_id == label_run1_id for s in samples1)

        samples2 = load_samples(db_session, label_run_id=label_run2.id)
        assert len(samples2) == 1
        assert samples2[0].label_run_id == label_run2.id

    def test_get_newest_label_run_id(self, db_session, sample_training_samples):
        """Test getting the newest label run ID"""
        label_run2 = LabelRun(snapshot_id=2, label_source="test2")
        label_run3 = LabelRun(snapshot_id=3, label_source="test3")
        db_session.add_all([label_run2, label_run3])
        db_session.commit()

        newest_id = get_newest_label_run_id(db_session)
        assert newest_id == label_run3.id

    def test_get_newest_label_run_id_empty(self, db_session):
        """Test getting newest label run ID when none exist"""
        newest_id = get_newest_label_run_id(db_session)
        assert newest_id is None


class TestSavePredictionsToDb:
    """Test save_predictions_to_db function"""

    def test_save_predictions(self, db_session, sample_training_samples):
        """Test saving predictions to database"""
        run = create_model_run(
            db_session,
            model_path="test_model",
            taxonomy="legacy",
            use_baseline=False,
            config={"test": True},
        )

        samples = sample_training_samples[:2]
        predictions = ["primary_author", "variant"]  # second is wrong
        confidences = [0.95, 0.82]
        probabilities = [
            {"primary_author": 0.95, "subject": 0.03, "variant": 0.02},
            {"primary_author": 0.10, "subject": 0.08, "variant": 0.82},
        ]

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

        saved = db_session.query(SamplePrediction).all()
        assert len(saved) == 2

        pred1 = saved[0]
        assert pred1.predicted_label == "primary_author"
        assert pred1.is_correct is True

        pred2 = saved[1]
        assert pred2.predicted_label == "variant"
        assert pred2.is_correct is False

    def test_save_predictions_without_labels(self, db_session):
        """Test saving predictions for unlabeled samples"""
        label_run = LabelRun(snapshot_id=1, label_source="test")
        db_session.add(label_run)
        db_session.flush()

        sample = TrainingSample(
            snapshot_id=1,
            node_id=100,
            label_run_id=label_run.id,
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

        run = create_model_run(
            db_session,
            model_path="test_model",
            taxonomy="legacy",
            use_baseline=False,
            config={},
        )

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
        assert run.run_type == "baseline"
        assert run.taxonomy == "v2"
        assert "baseline-v2" in run.model_version
        assert "setfit-baseline-v2" in run.base_model_id

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
        assert run.model_version == "/path/to/model"
        assert "setfit-training-v1" in run.base_model_id
