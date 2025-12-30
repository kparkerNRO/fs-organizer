"""Tests for train.py"""

import pytest
from fine_tuning.services.train import (
    TrainConfigSettings,
    augment_with_hard_negatives,
    prepare_training_data,
)
from storage.manager import StorageManager

from .factories import LabelRunFactory, TrainingSampleFactory


@pytest.mark.ml
class TestAugmentWithHardNegatives:
    """Test augment_with_hard_negatives function"""

    def test_basic_augmentation(self):
        """Test basic hard negative augmentation with similar items from different classes"""
        # Create asset_type and content_subject items with similar names (both contain "resolution")
        train_texts = [
            "High Resolution Images",
            "Low Resolution Images",
            "High Resolution Characters",
        ]
        train_leaf_keys = [
            "high_resolution_images",
            "low_resolution_images",
            "high_resolution_characters",
        ]
        train_labels = [
            0,
            0,
            1,
        ]  # First two are asset_type (0), last is content_subject (1)
        id2label = {0: "asset_type", 1: "content_subject"}
        confusable_labels = {"asset_type"}

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

        # Both asset_type items find the similar content_subject item as hard negative
        # 2 asset_type items × 1 hard negative × 1 factor = 2 pairs (anchor + hard negative)
        assert len(extra_texts) >= 2  # At least 2 hard negative pairs
        assert len(extra_labels) >= 2

    def test_no_similar_candidates(self):
        """Test when there are no similar candidates due to high threshold"""
        # Completely dissimilar strings
        train_texts = ["AAA", "BBB", "CCC"]
        train_leaf_keys = ["aaa", "bbb", "ccc"]
        train_labels = [0, 1, 2]
        id2label = {0: "asset_type", 1: "content_subject", 2: "other"}
        confusable_labels = {"asset_type"}

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
        id2label = {0: "asset_type", 1: "content_subject"}
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
        # 1 asset_type item and 4 similar content_subject items (all variations of "resolution")
        train_texts = [
            "Resolution A",
            "Resolution B",
            "Resolution C",
            "Resolution D",
            "Resolution E",
        ]
        train_leaf_keys = [
            "resolution_a",
            "resolution_b",
            "resolution_c",
            "resolution_d",
            "resolution_e",
        ]
        train_labels = [0, 1, 1, 1, 1]  # 1 asset_type (0), 4 content_subject (1)
        id2label = {0: "asset_type", 1: "content_subject"}
        confusable_labels = {"asset_type"}

        # Test with k=1: asset_type finds 1 similar content_subject
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

        # Test with k=3: asset_type finds up to 3 similar content_subject
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

        # k=1: 1 anchor + 1 hard negative = 2 samples
        assert len(extra_texts_k1) == 2
        # k=3: 1 anchor + 3 hard negatives = 4 samples (should find more than k=1)
        assert len(extra_texts_k3) == 4
        assert len(extra_texts_k3) > len(extra_texts_k1)

    def test_factor_parameter(self):
        """Test that factor parameter multiplies the augmentation amount"""
        # 2 asset_type items, 1 similar content_subject item
        train_texts = ["High Res Images", "Low Res Images", "High Res Characters"]
        train_leaf_keys = ["high_res_images", "low_res_images", "high_res_characters"]
        train_labels = [0, 0, 1]  # 2 asset_type (0), 1 content_subject (1)
        id2label = {0: "asset_type", 1: "content_subject"}
        confusable_labels = {"asset_type"}

        # Test with factor=1: each asset_type finds 1 content_subject hard negative × 1 factor
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

        # Test with factor=3: same pairs but repeated 3x
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

        # Factor=1: at least 1 asset_type finds the content_subject as hard negative = at least 2 samples
        assert len(extra_texts_f1) >= 2
        # Factor=3: same pairs but repeated 3x = at least 6 samples
        assert len(extra_texts_f3) >= 6
        # Factor=3 should be 3x factor=1
        assert len(extra_texts_f3) == 3 * len(extra_texts_f1)

    def test_empty_leaf_keys(self):
        """Test handling of empty leaf keys"""
        train_texts = ["Text1", "Text2", "Text3"]
        train_leaf_keys = ["", "text2", ""]  # Some empty
        train_labels = [0, 1, 0]
        id2label = {0: "asset_type", 1: "content_subject"}
        confusable_labels = {"asset_type"}

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


@pytest.mark.ml
class TestPrepareTrainingData:
    """Test prepare_training_data function"""

    def test_basic_data_preparation(self, label_run, tmp_path):
        """Test basic training data preparation"""
        # Create labeled training samples using factory
        samples = [
            TrainingSampleFactory.build(
                label_run_id=label_run.id,
                name_norm="high_res",
                label="asset_type",
            ),
            TrainingSampleFactory.build(
                label_run_id=label_run.id,
                name_norm="character_art",
                label="content_subject",
            ),
            TrainingSampleFactory.build(
                label_run_id=label_run.id,
                name_norm="environment",
                label="other",
            ),
            TrainingSampleFactory.build(
                label_run_id=label_run.id,
                name_norm="low_res",
                label="asset_type",
            ),
            TrainingSampleFactory.build(
                label_run_id=label_run.id,
                name_norm="portraits",
                label="content_subject",
            ),
            TrainingSampleFactory.build(
                label_run_id=label_run.id,
                name_norm="backgrounds",
                label="other",
            ),
        ]

        # Create manager
        manager = StorageManager(
            tmp_path,
            initialize_work=False,
            initialize_training=True,
        )

        # Add samples to manager's database
        with manager.get_training_session() as session:
            # Create label_run in this database first
            LabelRunFactory._meta.sqlalchemy_session = session
            db_label_run = LabelRunFactory.build(
                id=label_run.id, snapshot_id=label_run.snapshot_id
            )
            session.add(db_label_run)
            session.flush()  # Ensure label_run is created before samples

            # Add the pre-built samples directly
            for sample in samples:
                session.add(sample)
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
        assert len(train_ds) == 3
        assert len(test_ds) == 3
        assert len(train_ds) + len(test_ds) == 6

        # Check id2label mapping (all 6 v2 taxonomy labels)
        assert len(id2label) == 6
        assert all(isinstance(k, int) for k in id2label.keys())
        assert all(isinstance(v, str) for v in id2label.values())

    def test_with_hard_negatives(self, label_run, tmp_path):
        """Test data preparation with hard negative mining enabled"""
        samples = [
            TrainingSampleFactory.build(
                label_run_id=label_run.id,
                name_norm="high_resolution_images",
                label="asset_type",
                text="High Resolution Images",
            ),
            TrainingSampleFactory.build(
                label_run_id=label_run.id,
                name_norm="low_resolution_images",
                label="asset_type",
                text="Low Resolution Images",
            ),
            TrainingSampleFactory.build(
                label_run_id=label_run.id,
                name_norm="high_resolution_characters",
                label="content_subject",
                text="High Resolution Characters",
            ),
            TrainingSampleFactory.build(
                label_run_id=label_run.id,
                name_norm="low_resolution_characters",
                label="content_subject",
                text="Low Resolution Characters",
            ),
        ]

        manager = StorageManager(
            tmp_path,
            initialize_work=False,
            initialize_training=True,
        )

        with manager.get_training_session() as session:
            # Create label_run in this database first
            LabelRunFactory._meta.sqlalchemy_session = session
            db_label_run = LabelRunFactory.build(
                id=label_run.id, snapshot_id=label_run.snapshot_id
            )
            session.add(db_label_run)
            session.flush()  # Ensure label_run is created before samples

            # Add samples
            for sample in samples:
                session.add(sample)
            session.commit()

        config = TrainConfigSettings(
            test_size=0.33,
            seed=42,
            no_hard_negatives=False,
            hardneg_labels="asset_type",
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

        # With test_size=0.33 and 4 samples: typically 2-3 train, 1-2 test (stratified)
        # Hard negatives: asset_type and content_subject samples have similar names (both contain "resolution")
        # so confusable cross-class pairs should be found and added to training set
        assert len(test_ds) >= 1  # At least 1 test sample
        assert len(train_ds) >= 2  # At least 2 train samples (original split)
        # Hard negatives should increase train_ds size beyond original split
        assert len(train_ds) > len(test_ds)  # More training than test
        assert len(id2label) == 6  # All v2 taxonomy labels

    def test_no_labeled_samples(self, label_run, tmp_path):
        """Test error when no labeled samples are found"""
        sample = TrainingSampleFactory.build(label_run_id=label_run.id, label=None)

        manager = StorageManager(
            tmp_path,
            initialize_work=False,
            initialize_training=True,
        )

        with manager.get_training_session() as session:
            # Create label_run in this database first
            LabelRunFactory._meta.sqlalchemy_session = session
            db_label_run = LabelRunFactory.build(
                id=label_run.id, snapshot_id=label_run.snapshot_id
            )
            session.add(db_label_run)
            session.flush()  # Ensure label_run is created before samples

            # Add sample
            session.add(sample)
            session.commit()

        with pytest.raises(ValueError, match="No labeled training samples"):
            prepare_training_data(
                config=TrainConfigSettings(),
                manager=manager,
                taxonomy="v2",
                label_run_id=label_run.id,
            )

    def test_unknown_labels(self, label_run, tmp_path):
        """Test error when unknown labels are found"""
        sample = TrainingSampleFactory.build(
            label_run_id=label_run.id,
            label="invalid_label_xyz",
        )

        manager = StorageManager(
            tmp_path,
            initialize_work=False,
            initialize_training=True,
        )

        with manager.get_training_session() as session:
            # Create label_run in this database first
            LabelRunFactory._meta.sqlalchemy_session = session
            db_label_run = LabelRunFactory.build(
                id=label_run.id, snapshot_id=label_run.snapshot_id
            )
            session.add(db_label_run)
            session.flush()  # Ensure label_run is created before samples

            # Add sample
            session.add(sample)
            session.commit()

        with pytest.raises(ValueError, match="Unknown labels found"):
            prepare_training_data(
                config=TrainConfigSettings(),
                manager=manager,
                taxonomy="v2",
                label_run_id=label_run.id,
            )

    def test_invalid_taxonomy(self, label_run, tmp_path):
        """Test error with invalid taxonomy"""
        config = TrainConfigSettings()
        manager = StorageManager(
            tmp_path,
            initialize_work=False,
            initialize_training=True,
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
        self,
        training_session,
        label_run,
        tmp_path,
        test_size,
        expected_train,
        expected_test,
    ):
        """Test that test_size parameter affects split"""
        samples = [
            TrainingSampleFactory.build(
                label_run_id=label_run.id,
                name_norm=f"test{i}",
                label="asset_type" if i % 2 == 0 else "content_subject",
            )
            for i in range(10)
        ]

        manager = StorageManager(
            tmp_path,
            initialize_work=False,
            initialize_training=True,
        )

        with manager.get_training_session() as session:
            # Create label_run in this database first
            LabelRunFactory._meta.sqlalchemy_session = session
            db_label_run = LabelRunFactory.build(
                id=label_run.id, snapshot_id=label_run.snapshot_id
            )
            session.add(db_label_run)
            session.flush()  # Ensure label_run is created before samples

            # Add samples
            for sample in samples:
                session.add(sample)
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
