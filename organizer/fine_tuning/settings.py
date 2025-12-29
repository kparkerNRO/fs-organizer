"""Pydantic models for fine-tuning CLI settings."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class StorageSettings(BaseModel):
    """Settings for database storage paths."""

    storage_path: Path = Field(
        "data",
        description="Path to the main storage directory containing all databases.",
    )


class CommonSettings(StorageSettings):
    """Common settings for all commands."""

    taxonomy: str = Field(
        "legacy",
        description="Label taxonomy to use: v1, v2, or legacy",
    )
    label_run_id: Optional[int] = Field(
        None,
        description="Label run ID to use for training labels (defaults to newest)",
    )


class TrainSettings(BaseModel):
    """Settings for the 'train' command."""

    output_dir: Path = Field(
        "./leaf_classifier_setfit",
        description="Directory to save trained model",
    )
    model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Base sentence transformer model",
    )
    batch_size: int = Field(
        32,
        description="Batch size for training (must be multiple of samples_per_label)",
    )
    num_epochs: int = Field(
        6,
        description="Number of training epochs",
    )
    learning_rate: float = Field(
        2e-5,
        description="Learning rate",
    )
    samples_per_label: int = Field(
        2,
        description="Samples per label for triplet loss batching",
    )
    hardneg_k: int = Field(
        2,
        description="Number of hard negatives to mine per anchor",
    )
    hardneg_min_sim: float = Field(
        0.25,
        description="Minimum similarity threshold for hard negative mining",
    )
    hardneg_factor: int = Field(
        2,
        description="Oversampling factor for hard negatives",
    )
    hardneg_labels: str = Field(
        "",
        description="Comma-separated labels to mine hard negatives for (defaults to all confusable labels in taxonomy)",
    )
    test_size: float = Field(
        0.2,
        description="Fraction of data to use for testing (0.0-1.0)",
    )
    seed: int = Field(
        42,
        description="Random seed for reproducibility",
    )
    no_triplet_loss: bool = Field(
        False,
        description="Disable triplet loss (use default SetFit loss)",
    )
    no_hard_negatives: bool = Field(
        False,
        description="Disable hard negative mining and oversampling",
    )


class PredictSettings(BaseModel):
    """Settings for the 'predict' command."""

    model_path: Optional[Path] = Field(
        None,
        description="Path to fine-tuned SetFit model (required unless --use-baseline)",
    )
    use_baseline: bool = Field(
        False,
        description="Use baseline pre-trained model without fine-tuning",
    )
    split: Optional[str] = Field(
        None,
        description="Only evaluate on specific split: train, validation, or test",
    )
    labeled_only: bool = Field(
        False,
        description="Only run on samples with labels (for evaluation)",
    )
    save_predictions: bool = Field(
        False,
        description="Save predictions to database",
    )
    output_file: Optional[Path] = Field(
        None,
        description="Save predictions to CSV file",
    )


class ZeroShotSettings(BaseModel):
    """Settings for the 'zero-shot' command."""

    split: Optional[str] = Field(
        None,
        description="Only evaluate on specific split: train, validation, or test",
    )
    labeled_only: bool = Field(
        False,
        description="Only run on samples with labels (for evaluation)",
    )
    save_predictions: bool = Field(
        False,
        description="Save predictions to database",
    )
    output_file: Optional[Path] = Field(
        None,
        description="Save predictions to CSV file",
    )


class FeatureExtractionSettings(StorageSettings):
    """Settings for the 'extract-features' command."""

    snapshot_id: Optional[int] = Field(
        None,
        description="Snapshot ID to extract features from (defaults to highest snapshot_id if unset)",
    )
    batch_size: int = Field(
        1000,
        description="Number of samples to insert per batch",
    )


class GenerateSamplesSettings(StorageSettings):
    """Settings for the 'generate-samples' command."""

    output_csv: Path = Field(
        ...,
        description="Path to output CSV file for manual labeling",
    )
    snapshot_id: Optional[int] = Field(
        None,
        description="Snapshot ID to generate samples from (defaults to highest snapshot_id if unset)",
    )
    sample_size: int = Field(
        800,
        description="Number of samples to generate",
    )
    min_depth: int = Field(
        1,
        description="Minimum folder depth to sample from",
    )
    max_depth: int = Field(
        10,
        description="Maximum folder depth to sample from",
    )
    diversity_factor: float = Field(
        0.7,
        description="Balance between random and diverse sampling (0-1, higher=more diverse)",
    )
    use_heuristic: bool = Field(
        True,
        description="Include heuristic classifier predictions in CSV output",
    )
    heuristic_taxonomy: str = Field(
        "v2",
        description="Taxonomy for heuristic classifier (v1 or v2)",
    )


class ApplyClassificationsSettings(StorageSettings):
    """Settings for the 'apply-classifications' command."""

    input_csv: Path = Field(
        ...,
        description="CSV file with manual classifications",
    )
    labeler: str = Field(
        "manual",
        description="Name of the labeler (e.g., 'manual', 'human-v1')",
    )
    split: Optional[str] = Field(
        None,
        description="Data split: 'train', 'validation', or 'test'",
    )
    taxonomy: str = Field(
        "v2",
        description="Label taxonomy to validate against: v1, v2, or legacy",
    )
    validate_only: bool = Field(
        False,
        description="Only validate CSV without writing to database",
    )
