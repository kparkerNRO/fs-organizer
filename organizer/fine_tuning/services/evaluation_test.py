"""Tests for evaluation.py"""

import pytest

from fine_tuning.services.evaluation import evaluate_predictions
from fine_tuning.taxonomy import LABELS_LEGACY, LABELS_V1, LABELS_V2


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


class TestLabelTaxonomies:
    """Test label taxonomy definitions from evaluation context"""

    def test_v1_labels(self):
        """Verify V1 label taxonomy"""
        assert "person_or_group" in LABELS_V1
        assert len(LABELS_V1) == 6

    def test_v2_labels(self):
        """Verify V2 label taxonomy"""
        assert "creator_or_studio" in LABELS_V2
        assert len(LABELS_V2) == 6

    def test_legacy_labels(self):
        """Verify legacy label taxonomy"""
        assert "primary_author" in LABELS_LEGACY
        assert len(LABELS_LEGACY) == 8
