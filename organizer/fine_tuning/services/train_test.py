"""Tests for train.py"""

import pytest
from fine_tuning.services.train import (
    TrainConfigSettings,
    augment_with_hard_negatives,
    prepare_training_data,
)
from storage.manager import StorageManager
from storage.training_models import TrainingBase

from .factories import TrainingSampleFactory


class TestAugmentWithHardNegatives:
    """Test augment_with_hard_negatives function"""

    def test_basic_augmentation(self):
        """Test basic hard negative augmentation with similar items"""
        # Create similar variant items and dissimilar subject item
        train_texts = [
            "High Resolution Images",
            "Low Resolution Images",
            "Character Designs",
        ]
        train_leaf_keys = ["high_resolution_images", "low_resolution_images", "character_designs"]
        train_labels = [0, 0, 1]  # First two are same class (variant)
        id2label = {0: "variant", 1: "subject"}
        confusable_labels = {"variant"}

        extra_texts, extra_labels = augment_with_hard_negatives(
            train_texts=train_texts,
            train_leaf_keys=train_leaf_keys,
            train_labels=train_labels,
            id2label=id2label,
            confusable_labels=confusable_labels,
            k=1,
            min_sim=0.3,
            factor=1,
        )

        # Both variant items should find each other as hard negatives (2 pairs)
        # Each variant finds the other similar variant (k=1, factor=1)
        assert len(extra_texts) == 2
        assert len(extra_labels) == 2
        assert all(label == 0 for label in extra_labels)  # All should be variant class

    def test_no_similar_candidates(self):
        """Test when there are no similar candidates due to high threshold"""
        # Completely dissimilar strings
        train_texts = ["AAA", "BBB", "CCC"]
        train_leaf_keys = ["aaa", "bbb", "ccc"]
        train_labels = [0, 1, 2]
        id2label = {0: "variant", 1: "subject", 2: "other"}
        confusable_labels = {"variant"}

        extra_texts, extra_labels = augment_with_hard_negatives(
            train_texts=train_texts,
            train_leaf_keys=train_leaf_keys,
            train_labels=train_labels,
            id2label=id2label,
            confusable_labels=confusable_labels,
            k=2,
            min_sim=0.9,  # Very high threshold
            factor=2,
        )

        # Dissimilar strings + high threshold = no hard negatives
        assert len(extra_texts) == 0
        assert len(extra_labels) == 0

    def test_empty_confusable_labels(self):
        """Test with no confusable labels specified"""
        train_texts = ["Text1", "Text2"]
        train_leaf_keys = ["text1", "text2"]
        train_labels = [0, 1]
        id2label = {0: "variant", 1: "subject"}
        confusable_labels = set()  # Empty set

        extra_texts, extra_labels = augment_with_hard_negatives(
            train_texts=train_texts,
            train_leaf_keys=train_leaf_keys,
            train_labels=train_labels,
            id2label=id2label,
            confusable_labels=confusable_labels,
            k=2,
            min_sim=0.3,
            factor=2,
        )

        # Should generate nothing when no labels are confusable
        assert len(extra_texts) == 0
        assert len(extra_labels) == 0

    def test_k_parameter(self):
        """Test that k parameter limits number of hard negatives per sample"""
        # 4 similar variant items (all contain "Res"), 1 different subject
        train_texts = [
            "High Res",
            "Medium Res",
            "Low Res",
            "Ultra Res",
            "Character Art",
        ]
        train_leaf_keys = [
            "high_res",
            "medium_res",
            "low_res",
            "ultra_res",
            "character_art",
        ]
        train_labels = [0, 0, 0, 0, 1]
        id2label = {0: "variant", 1: "subject"}
        confusable_labels = {"variant"}

        # Test with k=1: each of 4 variants finds 1 similar neighbor = 4 total
        extra_texts_k1, extra_labels_k1 = augment_with_hard_negatives(
            train_texts=train_texts,
            train_leaf_keys=train_leaf_keys,
            train_labels=train_labels,
            id2label=id2label,
            confusable_labels=confusable_labels,
            k=1,
            min_sim=0.3,
            factor=1,
        )

        # Test with k=3: each of 4 variants finds up to 3 similar neighbors = 12 total
        extra_texts_k3, extra_labels_k3 = augment_with_hard_negatives(
            train_texts=train_texts,
            train_leaf_keys=train_leaf_keys,
            train_labels=train_labels,
            id2label=id2label,
            confusable_labels=confusable_labels,
            k=3,
            min_sim=0.3,
            factor=1,
        )

        # k=1: 4 variants × 1 neighbor = 4
        assert len(extra_texts_k1) == 4
        # k=3: 4 variants × 3 neighbors = 12
        assert len(extra_texts_k3) == 12

    def test_factor_parameter(self):
        """Test that factor parameter multiplies the augmentation amount"""
        # 2 similar variant items, 1 different subject
        train_texts = ["High Res Folder", "Low Res Folder", "Character Art"]
        train_leaf_keys = ["high_res_folder", "low_res_folder", "character_art"]
        train_labels = [0, 0, 1]
        id2label = {0: "variant", 1: "subject"}
        confusable_labels = {"variant"}

        # Test with factor=1: 2 variants × 1 neighbor × 1 factor = 2
        extra_texts_f1, extra_labels_f1 = augment_with_hard_negatives(
            train_texts=train_texts,
            train_leaf_keys=train_leaf_keys,
            train_labels=train_labels,
            id2label=id2label,
            confusable_labels=confusable_labels,
            k=1,
            min_sim=0.3,
            factor=1,
        )

        # Test with factor=3: 2 variants × 1 neighbor × 3 factor = 6
        extra_texts_f3, extra_labels_f3 = augment_with_hard_negatives(
            train_texts=train_texts,
            train_leaf_keys=train_leaf_keys,
            train_labels=train_labels,
            id2label=id2label,
            confusable_labels=confusable_labels,
            k=1,
            min_sim=0.3,
            factor=3,
        )

        # Factor=1: 2 variants find each other as neighbors = 2 hard negatives
        assert len(extra_texts_f1) == 2
        # Factor=3: same pairs but repeated 3x = 6 hard negatives
        assert len(extra_texts_f3) == 6

    def test_empty_leaf_keys(self):
        """Test handling of empty leaf keys"""
        train_texts = ["Text1", "Text2", "Text3"]
        train_leaf_keys = ["", "text2", ""]  # Some empty
        train_labels = [0, 1, 0]
        id2label = {0: "variant", 1: "subject"}
        confusable_labels = {"variant"}

        extra_texts, extra_labels = augment_with_hard_negatives(
            train_texts=train_texts,
            train_leaf_keys=train_leaf_keys,
            train_labels=train_labels,
            id2label=id2label,
            confusable_labels=confusable_labels,
            k=1,
            min_sim=0.3,
            factor=1,
        )

        # Should skip samples with empty leaf keys
        assert len(extra_texts) == len(extra_labels)


class TestPrepareTrainingData:
    """Test prepare_training_data function"""

    def test_basic_data_preparation(self, training_session, label_run, tmp_path):
        """Test basic training data preparation"""
        # Create labeled training samples using factory
        samples = [
            TrainingSampleFactory(
                label_run_id=label_run.id,
                name_norm="high_res",
                label="variant",
            ),
            TrainingSampleFactory(
                label_run_id=label_run.id,
                name_norm="character_art",
                label="subject",
            ),
            TrainingSampleFactory(
                label_run_id=label_run.id,
                name_norm="environment",
                label="other",
            ),
            TrainingSampleFactory(
                label_run_id=label_run.id,
                name_norm="low_res",
                label="variant",
            ),
        ]

        # Create manager
        manager = StorageManager(
            index_db_path=str(tmp_path / "index.db"),
            training_db_path=str(tmp_path / "test.db"),
        )

        # Initialize training database
        manager._init_training_db(TrainingBase)

        # Add samples to manager's database
        with manager.get_training_session() as session:
            for sample in samples:
                session.merge(sample)
            session.commit()

        # Create config
        config = TrainConfigSettings(
            test_size=0.5,
            seed=42,
            no_hard_negatives=True,
        )

        train_ds, test_ds, id2label = prepare_training_data(
            config=config,
            manager=manager,
            taxonomy="v2",
            label_run_id=label_run.id,
        )

        # Check that datasets were created with correct split (50% test_size)
        assert len(train_ds) == 2
        assert len(test_ds) == 2
        assert len(train_ds) + len(test_ds) == 4

        # Check id2label mapping (3 unique labels: variant, subject, other)
        assert len(id2label) == 3
        assert all(isinstance(k, int) for k in id2label.keys())
        assert all(isinstance(v, str) for v in id2label.values())

    def test_with_hard_negatives(self, training_session, label_run, tmp_path):
        """Test data preparation with hard negative mining enabled"""
        samples = [
            TrainingSampleFactory(
                label_run_id=label_run.id,
                name_norm="high_res",
                label="variant",
                text="High Resolution",
            ),
            TrainingSampleFactory(
                label_run_id=label_run.id,
                name_norm="low_res",
                label="variant",
                text="Low Resolution",
            ),
            TrainingSampleFactory(
                label_run_id=label_run.id,
                name_norm="character",
                label="subject",
                text="Character",
            ),
        ]

        manager = StorageManager(
            index_db_path=str(tmp_path / "index.db"),
            training_db_path=str(tmp_path / "test.db"),
        )

        manager._init_training_db(TrainingBase)

        with manager.get_training_session() as session:
            for sample in samples:
                session.merge(sample)
            session.commit()

        config = TrainConfigSettings(
            test_size=0.33,
            seed=42,
            no_hard_negatives=False,
            hardneg_labels="variant",
            hardneg_k=1,
            hardneg_min_sim=0.3,
            hardneg_factor=1,
        )

        train_ds, test_ds, id2label = prepare_training_data(
            config=config,
            manager=manager,
            taxonomy="v2",
            label_run_id=label_run.id,
        )

        # With test_size=0.33 and 3 samples: 2 train, 1 test
        # Train set has 2 original + 2 hard negatives (2 variant items find each other)
        assert len(train_ds) == 4
        assert len(test_ds) == 1
        assert len(id2label) == 2

    def test_no_labeled_samples(self, training_session, label_run, tmp_path):
        """Test error when no labeled samples are found"""
        sample = TrainingSampleFactory(label_run_id=label_run.id, label=None)

        manager = StorageManager(
            index_db_path=str(tmp_path / "index.db"),
            training_db_path=str(tmp_path / "test.db"),
        )

        manager._init_training_db(TrainingBase)

        with manager.get_training_session() as session:
            session.merge(sample)
            session.commit()

        with pytest.raises(ValueError, match="No labeled training samples"):
            prepare_training_data(
                config=TrainConfigSettings(),
                manager=manager,
                taxonomy="v2",
                label_run_id=label_run.id,
            )

    def test_unknown_labels(self, training_session, label_run, tmp_path):
        """Test error when unknown labels are found"""
        sample = TrainingSampleFactory(
            label_run_id=label_run.id,
            label="invalid_label_xyz",
        )

        manager = StorageManager(
            index_db_path=str(tmp_path / "index.db"),
            training_db_path=str(tmp_path / "test.db"),
        )

        manager._init_training_db(TrainingBase)

        with manager.get_training_session() as session:
            session.merge(sample)
            session.commit()

        with pytest.raises(ValueError, match="Unknown labels found"):
            prepare_training_data(
                config=TrainConfigSettings(),
                manager=manager,
                taxonomy="v2",
                label_run_id=label_run.id,
            )

    def test_invalid_taxonomy(self, training_session, label_run, tmp_path):
        """Test error with invalid taxonomy"""
        config = TrainConfigSettings()
        manager = StorageManager(
            index_db_path=str(tmp_path / "index.db"),
            training_db_path=str(tmp_path / "test.db"),
        )

        with pytest.raises(ValueError):
            prepare_training_data(
                config=config,
                manager=manager,
                taxonomy="invalid_taxonomy_xyz",
                label_run_id=label_run.id,
            )

    @pytest.mark.parametrize(
        "test_size,expected_train,expected_test",
        [
            (0.2, 8, 2),
            (0.5, 5, 5),
        ],
    )
    def test_test_size_parameter(
        self, training_session, label_run, tmp_path, test_size, expected_train, expected_test
    ):
        """Test that test_size parameter affects split"""
        samples = [
            TrainingSampleFactory(
                label_run_id=label_run.id,
                name_norm=f"test{i}",
                label="variant" if i % 2 == 0 else "subject",
            )
            for i in range(10)
        ]

        manager = StorageManager(
            index_db_path=str(tmp_path / "index.db"),
            training_db_path=str(tmp_path / "test.db"),
        )

        manager._init_training_db(TrainingBase)

        with manager.get_training_session() as session:
            for sample in samples:
                session.merge(sample)
            session.commit()

        config = TrainConfigSettings(
            test_size=test_size,
            seed=42,
            no_hard_negatives=True,
        )

        train_ds, test_ds, id2label = prepare_training_data(
            config=config,
            manager=manager,
            taxonomy="v2",
            label_run_id=label_run.id,
        )

        # With seed=42, splits are deterministic
        assert len(train_ds) == expected_train
        assert len(test_ds) == expected_test
        assert len(train_ds) + len(test_ds) == 10
