"""Model training database models.

This module defines SQLAlchemy models for the training.db database, which stores
training samples and labels for classifier fine-tuning.

This database is separate from work.db and is used specifically for the
model training/fine-tuning pipeline, not the classification pipeline.

IMPORTANT: Cross-database references (snapshot_id, node_id) to index.db are
validated at the application level, not by database constraints.
"""

from typing import List, Optional
from sqlalchemy import String, Float, Integer, Boolean, Index, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column

# Schema version (increment on breaking changes)
TRAINING_SCHEMA_VERSION = "1.0.0"


class TrainingBase(DeclarativeBase):
    pass


class TrainingRun(TrainingBase):
    """Training run for a classifier model.

    Each run represents one training session with specific hyperparameters.
    Multiple runs can be created to experiment with different configurations.
    """

    __tablename__ = "training_run"

    run_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    started_at: Mapped[str] = mapped_column(String)
    finished_at: Mapped[Optional[str]]
    status: Mapped[str] = mapped_column(
        String, default="running"
    )  # 'running' | 'completed' | 'failed' | 'cancelled'

    # Model and training configuration
    base_model_id: Mapped[Optional[str]]  # e.g., 'bert-base-uncased'
    model_version: Mapped[Optional[str]]
    hyperparameters_json: Mapped[Optional[str]]  # Learning rate, batch size, etc.
    config_hash: Mapped[Optional[str]]

    # Dataset info
    train_samples_count: Mapped[Optional[int]] = mapped_column(Integer)
    validation_samples_count: Mapped[Optional[int]] = mapped_column(Integer)
    test_samples_count: Mapped[Optional[int]] = mapped_column(Integer)

    # Final metrics
    final_train_loss: Mapped[Optional[float]] = mapped_column(Float)
    final_val_loss: Mapped[Optional[float]] = mapped_column(Float)
    final_val_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    final_val_f1: Mapped[Optional[float]] = mapped_column(Float)

    notes: Mapped[Optional[str]]

    # Relationships
    epochs: Mapped[List["TrainingEpoch"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    checkpoints: Mapped[List["ModelCheckpoint"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    predictions: Mapped[List["SamplePrediction"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class TrainingSample(TrainingBase):
    """Feature vector extracted from a node for model training.

    Stores extracted features for classifier fine-tuning.
    """

    __tablename__ = "training_sample"

    sample_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int]  # FK to index.db (cross-database)
    node_id: Mapped[int]  # FK to index.db (cross-database)

    # Raw context
    name_raw: Mapped[str] = mapped_column(String)
    name_norm: Mapped[str] = mapped_column(String)
    parent_name_norm: Mapped[Optional[str]]
    grandparent_name_norm: Mapped[Optional[str]]
    kind: Mapped[str] = mapped_column(String)  # 'file' | 'dir'
    file_source: Mapped[str] = mapped_column(String)
    depth: Mapped[int] = mapped_column(Integer)

    # Structural context (JSON arrays)
    child_names_topk_json: Mapped[Optional[str]] = mapped_column(String)
    sibling_names_topk_json: Mapped[Optional[str]] = mapped_column(String)
    descendant_file_exts_topk_json: Mapped[Optional[str]] = mapped_column(String)

    # Cue flags
    has_collab_cue: Mapped[bool] = mapped_column(Boolean, default=False)
    looks_like_format: Mapped[bool] = mapped_column(Boolean, default=False)
    child_has_media_type_cue: Mapped[bool] = mapped_column(Boolean, default=False)
    child_has_variant_hint: Mapped[bool] = mapped_column(Boolean, default=False)
    child_has_format_cue: Mapped[bool] = mapped_column(Boolean, default=False)
    sibling_has_variant_hint: Mapped[bool] = mapped_column(Boolean, default=False)

    # Model input text
    text: Mapped[str] = mapped_column(String)

    # Label (optional, can be added later)
    label: Mapped[Optional[str]]  # 'primary_author' | 'secondary_author' | etc.
    label_confidence: Mapped[Optional[float]] = mapped_column(Float)
    labeler: Mapped[Optional[str]]  # 'manual' | 'llm' | 'structural'

    # Data split
    split: Mapped[Optional[str]]  # 'train' | 'validation' | 'test'

    __table_args__ = (
        Index("idx_training_sample_snapshot", "snapshot_id"),
        Index("idx_training_sample_node", "node_id"),
        Index("idx_training_sample_snapshot_node", "snapshot_id", "node_id", unique=True),
        Index("idx_training_sample_split", "split"),
        Index("idx_training_sample_label", "label"),
    )


class TrainingEpoch(TrainingBase):
    """Metrics for a single training epoch.

    Tracks loss, accuracy, and other metrics for each epoch during training.
    """

    __tablename__ = "training_epoch"

    epoch_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("training_run.run_id"))
    epoch_number: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[Optional[str]]

    # Training metrics
    train_loss: Mapped[Optional[float]] = mapped_column(Float)
    train_accuracy: Mapped[Optional[float]] = mapped_column(Float)

    # Validation metrics
    val_loss: Mapped[Optional[float]] = mapped_column(Float)
    val_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    val_precision: Mapped[Optional[float]] = mapped_column(Float)
    val_recall: Mapped[Optional[float]] = mapped_column(Float)
    val_f1: Mapped[Optional[float]] = mapped_column(Float)

    # Per-class metrics (JSON)
    class_metrics_json: Mapped[Optional[str]]  # {'variant': {'precision': 0.9, ...}, ...}

    # Training info
    learning_rate: Mapped[Optional[float]] = mapped_column(Float)
    samples_processed: Mapped[Optional[int]] = mapped_column(Integer)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)

    # Relationship
    run: Mapped["TrainingRun"] = relationship(back_populates="epochs")

    __table_args__ = (
        Index("idx_training_epoch_run", "run_id"),
        Index("idx_training_epoch_run_number", "run_id", "epoch_number", unique=True),
    )


class ModelCheckpoint(TrainingBase):
    """Saved model checkpoint from a training run.

    Stores reference to saved model files and associated metadata.
    """

    __tablename__ = "model_checkpoint"

    checkpoint_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("training_run.run_id"))
    epoch_number: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[str] = mapped_column(String)

    # Checkpoint info
    checkpoint_path: Mapped[str] = mapped_column(String)  # Path to saved model
    checkpoint_type: Mapped[str] = mapped_column(
        String
    )  # 'best' | 'final' | 'periodic'
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer)

    # Metrics at checkpoint time
    val_loss: Mapped[Optional[float]] = mapped_column(Float)
    val_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    val_f1: Mapped[Optional[float]] = mapped_column(Float)

    # Metadata
    is_best: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[Optional[str]]

    # Relationship
    run: Mapped["TrainingRun"] = relationship(back_populates="checkpoints")

    __table_args__ = (Index("idx_model_checkpoint_run", "run_id"),)


class SamplePrediction(TrainingBase):
    """Prediction result for a training sample.

    Stores model predictions on samples for evaluation and error analysis.
    """

    __tablename__ = "sample_prediction"

    prediction_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("training_run.run_id"))
    sample_id: Mapped[int]  # FK to training_sample (same database)
    epoch_number: Mapped[Optional[int]] = mapped_column(
        Integer
    )  # Which epoch (None = final)

    # Prediction
    predicted_label: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float)
    probabilities_json: Mapped[Optional[str]]  # All class probabilities

    # Ground truth (for comparison)
    true_label: Mapped[Optional[str]]
    is_correct: Mapped[Optional[bool]] = mapped_column(Boolean)

    # For analysis
    prediction_type: Mapped[Optional[str]]  # 'train' | 'validation' | 'test'

    # Relationship
    run: Mapped["TrainingRun"] = relationship(back_populates="predictions")

    __table_args__ = (
        Index("idx_sample_prediction_run", "run_id"),
        Index("idx_sample_prediction_sample", "sample_id"),
        Index("idx_sample_prediction_run_sample", "run_id", "sample_id"),
    )


class Meta(TrainingBase):
    """Metadata key-value store for training.db.

    Used for storing schema_version and other database-level metadata.
    """

    __tablename__ = "meta"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[Optional[str]]
