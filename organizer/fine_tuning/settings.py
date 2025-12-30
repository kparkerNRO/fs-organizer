"""Pydantic models for fine-tuning CLI settings."""

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageSettings(BaseSettings):
    """Settings for database storage paths."""

    model_config = SettingsConfigDict(env_prefix="FS_ORGANIZER_")

    base_path: Path = Field(
        Path("."),
        description="Base directory for storage and model paths.",
    )

    storage_path: Path = Field(
        "data",
        description="Path to the main storage directory containing all databases.",
    )

    model_path: Path = Field(
        "models",
        description="Path to the directory containing trained models.",
    )

    taxonomy: str = Field(
        "v2",
        description="Label taxonomy to use: v1, v2, or legacy",
    )

    @model_validator(mode="after")
    def _apply_base_path(self) -> "StorageSettings":
        self.storage_path = self._resolve_under_base(self.storage_path)
        self.model_path = self._resolve_under_base(self.model_path)
        return self

    def _resolve_under_base(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self.base_path / path
