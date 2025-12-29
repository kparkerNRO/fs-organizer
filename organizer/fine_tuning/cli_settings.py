"""Pydantic settings for fine-tuning CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator


class TrainSettings(BaseModel):
    training_db: Path
    label_run_id: int | None = None
    taxonomy: Literal["v1", "v2", "legacy"] = "legacy"
    output_dir: Path = Path("./leaf_classifier_setfit")
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    num_epochs: int = 6
    learning_rate: float = 2e-5
    samples_per_label: int = 2
    hardneg_k: int = 2
    hardneg_min_sim: float = 0.25
    hardneg_factor: int = 2
    hardneg_labels: str = ""
    test_size: float = 0.2
    seed: int = 42
    no_triplet_loss: bool = False


class PredictionSettings(BaseModel):
    training_db: Path
    taxonomy: Literal["v1", "v2", "legacy"] = "legacy"
    split: Literal["train", "validation", "test"] | None = None
    labeled_only: bool = False
    save_predictions: bool = False
    output_file: Path | None = None
    label_run_id: int | None = None


class PredictSettings(PredictionSettings):
    model_path: Path | None = None
    use_baseline: bool = False

    @model_validator(mode="after")
    def validate_model_path(self) -> "PredictSettings":
        if not self.use_baseline and not self.model_path:
            raise ValueError("--model-path is required unless --use-baseline is set")
        return self


class ZeroShotSettings(PredictionSettings):
    taxonomy: Literal["v1", "v2", "legacy"] = "v2"


class ExtractFeaturesSettings(BaseModel):
    index_db: Path
    training_db: Path
    snapshot_id: int | None = None
    batch_size: int = 1000


class GenerateSamplesSettings(BaseModel):
    index_db: Path
    output_csv: Path
    snapshot_id: int | None = None
    sample_size: int = 800
    min_depth: int = 1
    max_depth: int = 10
    diversity_factor: float = 0.7
    use_heuristic: bool = True
    heuristic_taxonomy: Literal["v1", "v2"] = "v2"


class SelectDataSettings(BaseModel):
    output_csv: Path
    storage_path: Path | None = None
    snapshot_id: int | None = None
    sample_size: int = 200
    min_depth: int = 1
    max_depth: int = 10
    diversity_factor: float = 0.7


class ApplyClassificationsSettings(BaseModel):
    input_csv: Path
    storage_path: Path | None = None
    training_db_path: Path | None = None
    labeler: str = "manual"
    split: Literal["train", "validation", "test"] | None = None
    taxonomy: Literal["v1", "v2", "legacy"] = "v2"
    validate_only: bool = False
