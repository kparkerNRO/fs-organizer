"""Tests for the training service."""

import pytest
from unittest.mock import MagicMock
from contextlib import contextmanager

from storage.factories import TrainingSampleFactory
from fine_tuning.services.train import (
    TrainConfigSettings,
    augment_with_hard_negatives,
    prepare_training_data,
)
from fine_tuning.taxonomy import LABELS_V2  # Added import for LABELS_V2


class TestAugmentWithHardNegatives:
    """Tests for the augment_with_hard_negatives function."""

    def test_finds_negatives(self):
        """Test that hard negatives are found and added."""
        train_texts = ["text_a1", "text_b1", "text_c1"]
        train_leaf_keys = ["file management", "file manipulation", "photo editing"]
        train_labels = [0, 1, 2]
        id2label = {0: "manage", 1: "manipulate", 2: "photo"}
        confusable_labels = {"manage"}

        extra_texts, extra_labels = augment_with_hard_negatives(
            train_texts=train_texts,
            train_leaf_keys=train_leaf_keys,
            train_labels=train_labels,
            id2label=id2label,
            confusable_labels=confusable_labels,
            k=1,
            min_sim=0.2,
            factor=1,
        )

        # "file management" is similar to "file manipulation", so we get one hard negative pair
        assert extra_texts == ["text_a1", "text_b1"]
        assert extra_labels == [0, 1]

    def test_respects_k_and_factor(self):
        """Test that k and factor parameters are respected."""
        train_texts = ["text_a1", "text_b1", "text_b2", "text_c1"]
        train_leaf_keys = [
            "content creation tools",
            "content generation tools",
            "content editing tools",
            "system utilities",
        ]
        train_labels = [0, 1, 1, 2]
        id2label = {0: "create", 1: "generate_edit", 2: "system"}
        confusable_labels = {"create"}

        extra_texts, extra_labels = augment_with_hard_negatives(
            train_texts=train_texts,
            train_leaf_keys=train_leaf_keys,
            train_labels=train_labels,
            id2label=id2label,
            confusable_labels=confusable_labels,
            k=2,
            min_sim=0.2,
            factor=2,
        )

        # We expect 1 anchor ("create") to find 2 hard negatives.
        # This block (anchor, neg1, neg2) is then added 'factor' times.
        assert len(extra_texts) == 2 * (1 + 2)  # factor * (1 anchor + k negatives)
        assert len(extra_labels) == 6
        assert extra_texts.count("text_a1") == 2  # anchor text
        assert extra_labels.count(0) == 2  # anchor label
        assert extra_labels.count(1) == 4  # negative labels

    def test_no_confusable_labels(self):
        """Test that no samples are added if confusable_labels is empty."""
        train_texts = ["text_a1", "text_b1"]
        train_leaf_keys = ["file management", "file manipulation"]
        train_labels = [0, 1]
        id2label = {0: "manage", 1: "manipulate"}
        confusable_labels = set()

        extra_texts, extra_labels = augment_with_hard_negatives(
            train_texts, train_leaf_keys, train_labels, id2label, confusable_labels
        )
        assert not extra_texts
        assert not extra_labels

    def test_min_sim_threshold(self):
        """Test that min_sim threshold is respected."""
        train_texts = ["text_a1", "text_b1"]
        train_leaf_keys = ["apples and oranges", "boats and planes"]
        train_labels = [0, 1]
        id2label = {0: "fruit", 1: "vehicles"}
        confusable_labels = {"fruit"}

        extra_texts, extra_labels = augment_with_hard_negatives(
            train_texts=train_texts,
            train_leaf_keys=train_leaf_keys,
            train_labels=train_labels,
            id2label=id2label,
            confusable_labels=confusable_labels,
            min_sim=0.9,  # High threshold, no match expected
        )
        assert not extra_texts
        assert not extra_labels

    def test_empty_leaf_keys(self):
        """Test that samples with empty leaf keys are skipped."""
        train_texts = ["Text1", "Text2", "Text3"]
        train_leaf_keys = ["", "text2", "zzzz"]  # First sample has empty leaf key
        train_labels = [0, 1, 0]
        id2label = {0: "label_a", 1: "label_b"}
        confusable_labels = {"label_a"}

        extra_texts, extra_labels = augment_with_hard_negatives(
            train_texts=train_texts,
            train_leaf_keys=train_leaf_keys,
            train_labels=train_labels,
            id2label=id2label,
            confusable_labels=confusable_labels,
        )

        # The anchor with the empty leaf key should be skipped. The other anchor ('zzzz') has no
        # similar items from other labels ('text2') to be paired with.
        assert not extra_texts
        assert not extra_labels


class FakeStorageManager:
    """A fake StorageManager that uses an in-memory database session."""

    def __init__(self, session):
        self._session = session

    @contextmanager
    def get_training_session(self):
        yield self._session

    def get_training_db_path(self):
        return "in-memory-fake.db"


@pytest.fixture
def fake_storage_manager(training_session):
    """Fixture to create a FakeStorageManager."""
    return FakeStorageManager(training_session)


@pytest.fixture
def populated_training_session(training_session, label_run):
    # Use real v2 labels
    TrainingSampleFactory.create_batch(
        10, label_run_id=label_run.id, label="asset_type"
    )
    TrainingSampleFactory.create_batch(
        10, label_run_id=label_run.id, label="content_subject"
    )
    training_session.commit()
    return training_session


class TestPrepareTrainingData:
    """Tests for the prepare_training_data function."""

    def test_happy_path(
        self, fake_storage_manager, populated_training_session, label_run
    ):
        """Test successful preparation of training and test datasets."""
        config = TrainConfigSettings(no_hard_negatives=True, test_size=0.2, seed=42)
        # Use the real get_labels function
        train_ds, test_ds, id2label = prepare_training_data(
            config,
            fake_storage_manager,
            "v2",  # Use a real taxonomy
            label_run.id,
        )

        assert len(train_ds) == 16  # 80% of 20
        assert len(test_ds) == 4  # 20% of 20
        # Assert id2label reflects the actual V2 labels sorted alphabetically
        expected_labels_sorted = sorted(list(LABELS_V2))
        expected_id2label = {i: label for i, label in enumerate(expected_labels_sorted)}
        assert id2label == expected_id2label
        # Verify stratification - counts based on expected label mapping
        # "asset_type" is the first label alphabetically, "content_subject" is the second
        assert test_ds["label"].count(expected_labels_sorted.index("asset_type")) == 2
        assert (
            test_ds["label"].count(expected_labels_sorted.index("content_subject")) == 2
        )

    def test_no_samples(self, fake_storage_manager, training_session, label_run):
        """Test ValueError when no labeled samples are found."""
        config = TrainConfigSettings()
        with pytest.raises(ValueError, match="No labeled training samples found"):
            prepare_training_data(
                config,
                fake_storage_manager,
                "v2",  # Use a real taxonomy
                label_run.id,
            )

    def test_unknown_labels(self, fake_storage_manager, training_session, label_run):
        """Test ValueError when samples have labels not in the taxonomy."""
        # Use a label not in v2 taxonomy
        TrainingSampleFactory(
            label_run_id=label_run.id, label="unknown_label_not_in_v2"
        )
        training_session.commit()

        config = TrainConfigSettings()
        with pytest.raises(ValueError, match="Unknown labels found"):
            prepare_training_data(
                config,
                fake_storage_manager,
                "v2",  # Use a real taxonomy
                label_run.id,
            )

    def test_hard_negatives_disabled(
        self, fake_storage_manager, populated_training_session, label_run
    ):
        """Test that hard negative mining is skipped when disabled."""
        config = TrainConfigSettings(no_hard_negatives=True, test_size=0.5, seed=42)
        mock_augment = MagicMock()

        train_ds, _, _ = prepare_training_data(
            config=config,
            manager=fake_storage_manager,
            taxonomy="v2",  # Use a real taxonomy
            label_run_id=label_run.id,
            augment_fn=mock_augment,
        )

        mock_augment.assert_not_called()
        assert len(train_ds) == 10  # 50% of 20

    def test_hard_negatives_enabled(
        self, fake_storage_manager, training_session, label_run
    ):
        """Test that hard negative mining is called when enabled, using the v2 taxonomy."""
        # Use valid v2 labels for the samples
        TrainingSampleFactory.create_batch(
            5, label_run_id=label_run.id, label="asset_type"
        )
        TrainingSampleFactory.create_batch(
            5, label_run_id=label_run.id, label="content_subject"
        )
        training_session.commit()

        config = TrainConfigSettings(
            no_hard_negatives=False,
            hardneg_labels="asset_type",  # A valid V2 label
            test_size=0.5,
            seed=42,
        )
        # Mock the augmentation function to test that it's called correctly
        mock_augment = MagicMock()
        mock_augment.return_value = (["extra_text_1", "extra_text_2"], [0, 1])

        # Original train size is 50% of 10 = 5.
        # The mocked augmentation adds 2 samples. Expected final size = 7.
        train_ds, _, _ = prepare_training_data(
            config=config,
            manager=fake_storage_manager,
            taxonomy="v2",
            label_run_id=label_run.id,
            augment_fn=mock_augment,
        )

        mock_augment.assert_called_once()
        assert len(train_ds) == 7

    def test_invalid_taxonomy(self, fake_storage_manager, label_run):
        """Test that a ValueError is raised for an invalid taxonomy."""
        config = TrainConfigSettings()
        # The real `get_labels` is expected to raise an error for a non-existent taxonomy
        with pytest.raises(ValueError, match="Unknown taxonomy"):
            prepare_training_data(
                config=config,
                manager=fake_storage_manager,
                taxonomy="invalid-taxonomy-name",
                label_run_id=label_run.id,
            )
