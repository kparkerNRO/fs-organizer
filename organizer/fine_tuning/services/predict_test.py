"""Tests for predict.py - REFACTORED VERSION using factories

This demonstrates the factory pattern applied to predict_test.py.
Compare with predict_test.py to see the improvements.
"""

import json

import pytest
from fine_tuning.services.predict import (
    create_and_save_run_results,
    create_model_run,
    save_predictions_to_db,
)
from storage.training_models import ModelRun, SamplePrediction

from .factories import TrainingSampleFactory


@pytest.mark.ml
class TestSavePredictionsToDb:
    """Test save_predictions_to_db function"""

    def test_save_basic_predictions(self, training_session, label_run, model_run):
        """Test saving predictions to database"""
        # Create samples using factory - only specify what matters
        samples = [
            TrainingSampleFactory(label_run_id=label_run.id, label="asset_type"),
            TrainingSampleFactory(label_run_id=label_run.id, label="content_subject"),
        ]

        predictions = ["asset_type", "other"]
        confidences = [0.95, 0.75]
        probabilities = [
            {"asset_type": 0.95, "content_subject": 0.05},
            {"other": 0.75, "asset_type": 0.25},
        ]

        num_saved = save_predictions_to_db(
            session=training_session,
            samples=samples,
            predictions=predictions,
            confidences=confidences,
            probabilities=probabilities,
            run_id=1,
            prediction_type="test",
        )

        assert num_saved == 2

        # Verify predictions were saved
        saved_predictions = training_session.query(SamplePrediction).all()
        assert len(saved_predictions) == 2

        # Check first prediction
        pred1 = saved_predictions[0]
        assert pred1.predicted_label == "asset_type"
        assert pred1.confidence == 0.95
        assert pred1.true_label == "asset_type"
        assert pred1.is_correct is True
        assert pred1.prediction_type == "test"

        # Check second prediction
        pred2 = saved_predictions[1]
        assert pred2.predicted_label == "other"
        assert pred2.confidence == 0.75
        assert pred2.true_label == "content_subject"
        assert pred2.is_correct is False

    def test_save_predictions_with_list_probabilities(
        self, training_session, label_run, model_run
    ):
        """Test saving predictions with list-format probabilities"""
        sample = TrainingSampleFactory(label_run_id=label_run.id, label="asset_type")

        num_saved = save_predictions_to_db(
            session=training_session,
            samples=[sample],
            predictions=["asset_type"],
            confidences=[0.95],
            probabilities=[[0.95, 0.03, 0.02]],  # List format
            run_id=1,
        )

        assert num_saved == 1

        saved_prediction = training_session.query(SamplePrediction).first()
        probs_dict = json.loads(saved_prediction.probabilities_json)

        # Should convert list to dict with label_N keys
        assert set(probs_dict.keys()) == {"label_0", "label_1", "label_2"}
        assert probs_dict["label_0"] == 0.95

    def test_save_predictions_unlabeled_samples(
        self, training_session, label_run, model_run
    ):
        """Test saving predictions for unlabeled samples"""
        sample = TrainingSampleFactory(label_run_id=label_run.id, label=None)

        num_saved = save_predictions_to_db(
            session=training_session,
            samples=[sample],
            predictions=["asset_type"],
            confidences=[0.8],
            probabilities=[{"asset_type": 0.8, "content_subject": 0.2}],
            run_id=1,
        )

        assert num_saved == 1

        saved_prediction = training_session.query(SamplePrediction).first()
        assert saved_prediction.predicted_label == "asset_type"
        assert saved_prediction.true_label is None
        assert saved_prediction.is_correct is None  # Can't determine correctness

    @pytest.mark.parametrize(
        "prediction_type",
        ["train", "validation", "test", "all"],
    )
    def test_prediction_type_parameter(
        self, training_session, label_run, model_run, prediction_type
    ):
        """Test different prediction type values"""
        sample = TrainingSampleFactory(label_run_id=label_run.id)

        save_predictions_to_db(
            session=training_session,
            samples=[sample],
            predictions=["asset_type"],
            confidences=[0.95],
            probabilities=[{"asset_type": 0.95}],
            run_id=1,
            prediction_type=prediction_type,
        )

        saved_prediction = training_session.query(SamplePrediction).first()
        assert saved_prediction.prediction_type == prediction_type


@pytest.mark.ml
class TestCreateModelRun:
    """Test create_model_run function"""

    def test_create_baseline_run(self, training_session, label_run):
        """Test creating a baseline model run"""
        config = {"model": "test-model", "batch_size": 32}

        run = create_model_run(
            session=training_session,
            model_path=None,
            taxonomy="v2",
            use_baseline=True,
            config=config,
        )

        assert run.run_id is not None
        assert run.status == "running"
        assert run.run_type == "baseline"
        assert run.base_model_id == "setfit-baseline-v2"
        assert run.model_version == "baseline-v2"
        assert run.model_type == "setfit"
        assert run.taxonomy == "v2"
        assert run.started_at is not None

        # Check config was saved
        saved_config = json.loads(run.hyperparameters_json)
        assert saved_config["model"] == "test-model"
        assert saved_config["batch_size"] == 32

    def test_create_finetuned_run(self, training_session, label_run):
        """Test creating a fine-tuned model run"""
        config = {"learning_rate": 2e-5, "num_epochs": 4}
        model_path = "/path/to/model"

        run = create_model_run(
            session=training_session,
            model_path=model_path,
            taxonomy="v1",
            use_baseline=False,
            config=config,
        )

        assert run.run_type == "evaluation"
        assert run.base_model_id == "setfit-evaluation-v1"
        assert run.model_version == model_path
        assert run.taxonomy == "v1"

    def test_create_training_run(self, training_session, label_run):
        """Test creating a training run with explicit run_type"""
        config = {"learning_rate": 2e-5}

        run = create_model_run(
            session=training_session,
            model_path="/path/to/model",
            taxonomy="v2",
            use_baseline=False,
            config=config,
            run_type="training",
        )

        assert run.run_type == "training"
        assert run.base_model_id == "setfit-training-v2"

    def test_create_run_with_training_data_source(self, training_session, label_run):
        """Test creating run with training data source"""
        config = {}

        run = create_model_run(
            session=training_session,
            model_path=None,
            taxonomy="v2",
            use_baseline=True,
            config=config,
            training_data_source="manual-labels-2024",
        )

        assert run.training_data_source == "manual-labels-2024"

    def test_auto_detect_run_type(self, training_session, label_run):
        """Test automatic run type detection"""
        config = {}

        # Baseline should auto-detect to "baseline"
        run_baseline = create_model_run(
            session=training_session,
            model_path=None,
            taxonomy="v2",
            use_baseline=True,
            config=config,
            run_type=None,
        )
        assert run_baseline.run_type == "baseline"

        # Non-baseline should auto-detect to "evaluation"
        run_eval = create_model_run(
            session=training_session,
            model_path="/path",
            taxonomy="v2",
            use_baseline=False,
            config=config,
            run_type=None,
        )
        assert run_eval.run_type == "evaluation"


@pytest.mark.ml
class TestCreateAndSaveRunResults:
    """Test create_and_save_run_results function"""

    def test_save_results_with_metrics(self, training_session, label_run):
        """Test saving run results with evaluation metrics"""

        # Create a simple mock classifier class
        class MockClassifier:
            labels = ["asset_type", "content_subject", "other"]

        config_dict = {"model": "test", "batch_size": 32}
        classifier = MockClassifier()

        # Use factories to create samples
        samples = TrainingSampleFactory.create_batch(2, label="asset_type")

        predictions = ["asset_type", "content_subject"]
        confidences = [0.95, 0.85]
        probabilities = [
            {"asset_type": 0.95, "content_subject": 0.05},
            {"content_subject": 0.85, "asset_type": 0.15},
        ]
        metrics = {
            "accuracy": 1.0,
            "macro_f1": 0.98,
            "weighted_f1": 0.99,
        }

        create_and_save_run_results(
            session=training_session,
            config_dict=config_dict,
            classifier=classifier,
            samples=samples,
            predictions=predictions,
            confidences=confidences,
            probabilities=probabilities,
            metrics=metrics,
            taxonomy="v2",
            model_path=None,
            use_baseline=True,
            split="test",
        )

        # Check run was created and completed
        run = training_session.query(ModelRun).first()
        assert run is not None
        assert run.status == "completed"
        assert run.finished_at is not None
        assert run.test_samples_count == 2
        assert run.final_val_accuracy == 1.0
        assert run.final_val_f1 == 0.98
        assert "Accuracy: 1.0000" in run.notes
        assert "Macro-F1: 0.9800" in run.notes

        # Check predictions were saved
        predictions_saved = training_session.query(SamplePrediction).all()
        assert len(predictions_saved) == 2

    def test_save_results_without_metrics(self, training_session, label_run):
        """Test saving run results without evaluation metrics"""

        class MockClassifier:
            labels = ["asset_type", "content_subject"]

        config_dict = {"model": "test"}
        classifier = MockClassifier()

        # Factory creates unlabeled sample by default or we can specify
        sample = TrainingSampleFactory(label_run_id=label_run.id, label=None)

        create_and_save_run_results(
            session=training_session,
            config_dict=config_dict,
            classifier=classifier,
            samples=[sample],
            predictions=["asset_type"],
            confidences=[0.8],
            probabilities=[{"asset_type": 0.8}],
            metrics={},  # No metrics
            taxonomy="v1",
            model_path="/path/to/model",
            use_baseline=False,
            split=None,
        )

        run = training_session.query(ModelRun).first()
        assert run is not None
        assert run.final_val_accuracy is None
        assert run.final_val_f1 is None
        # Notes should not contain metrics
        assert "Accuracy:" not in run.notes

    def test_baseline_run_type(self, training_session, label_run):
        """Test that baseline flag overrides run type"""

        class MockClassifier:
            labels = ["asset_type"]

        sample = TrainingSampleFactory(
            label_run_id=label_run.id,
        )

        create_and_save_run_results(
            session=training_session,
            config_dict={},
            classifier=MockClassifier(),
            samples=[sample],
            predictions=["asset_type"],
            confidences=[0.9],
            probabilities=[{"asset_type": 0.9}],
            metrics={},
            taxonomy="v2",
            model_path=None,
            use_baseline=True,
            split="validation",
        )

        run = training_session.query(ModelRun).first()
        assert "baseline" in run.notes
        assert run.run_type == "baseline"
