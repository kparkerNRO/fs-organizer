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
