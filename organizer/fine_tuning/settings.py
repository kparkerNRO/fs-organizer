"""Pydantic models for fine-tuning CLI settings."""

from pathlib import Path

from pydantic import BaseModel, Field


class StorageSettings(BaseModel):
    """Settings for database storage paths."""

    storage_path: Path = Field(
        "data",
        description="Path to the main storage directory containing all databases.",
    )

    model_path: Path = Field(
        "models",
        description="Path to the main storage directory containing all databases.",
    )

    taxonomy: str = Field(
        "v2",
        description="Label taxonomy to use: v1, v2, or legacy",
    )
