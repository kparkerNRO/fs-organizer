"""Work database models for pipeline processing.

This module defines SQLAlchemy models for the work.db database, which stores
intermediary pipeline processing results keyed by snapshot_id + run_id.

IMPORTANT: Cross-database references (snapshot_id, node_id) to index.db are
validated at the application level, not by database constraints.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import Float, ForeignKey, Index, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from storage.db_types import DateTime, JsonDict, JsonList

# Schema version (increment on breaking changes)
WORK_SCHEMA_VERSION = "1.0.0"


class WorkBase(DeclarativeBase):
    pass


class Run(WorkBase):
    """Processing run for a specific snapshot.

    Each run represents one execution of the pipeline on a snapshot.
    Multiple runs can be created for the same snapshot to experiment with
    different parameters or pipeline versions.
    """

    __tablename__ = "run"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int]  # FK to index.db (cross-database)
    started_at: Mapped[datetime] = mapped_column(DateTime)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(
        String, default="running"
    )  # 'running' | 'completed' | 'failed' | 'cancelled'
    pipeline_version: Mapped[Optional[str]]
    config_hash: Mapped[Optional[str]]
    model_id: Mapped[Optional[str]]
    notes: Mapped[Optional[str]]

    # Relationships
    stages: Mapped[List["StageState"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    group_iterations: Mapped[List["GroupIteration"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class StageState(WorkBase):
    """Tracks completion state of pipeline stages.

    IMPORTANT: snapshot_id must be validated against run.snapshot_id in
    mark_stage_complete() to prevent inconsistencies.
    """

    __tablename__ = "stage_state"

    stage_name: Mapped[str] = mapped_column(String, primary_key=True)
    snapshot_id: Mapped[int] = mapped_column(
        primary_key=True
    )  # Redundant with run.snapshot_id for query performance
    run_id: Mapped[int] = mapped_column(ForeignKey("run.id"), primary_key=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    input_fingerprint: Mapped[Optional[str]]
    output_fingerprint: Mapped[Optional[str]]

    # Relationship
    run: Mapped["Run"] = relationship(back_populates="stages")


class GroupIteration(WorkBase):
    """Grouping iteration within a run.

    Supports multiple grouping iterations on the same snapshot/run for
    iterative refinement or experimentation.

    IMPORTANT: snapshot_id must be validated against run.snapshot_id in
    create_group_iteration() to prevent inconsistencies.
    """

    __tablename__ = "stg_group_iteration"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run.id"))
    snapshot_id: Mapped[int]  # Redundant with run.snapshot_id for queries
    timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime)
    description: Mapped[Optional[str]]
    parameters: Mapped[Optional[dict]] = mapped_column(JsonDict)

    # Relationships
    run: Mapped["Run"] = relationship(back_populates="group_iterations")
    entries: Mapped[List["GroupEntry"]] = relationship(
        back_populates="iteration", cascade="all, delete-orphan"
    )
    categories: Mapped[List["GroupCategory"]] = relationship(
        back_populates="iteration", cascade="all, delete-orphan"
    )


class GroupEntry(WorkBase):
    """Entry for a node in a grouping iteration.

    IMPORTANT: node_id references index.db and must be validated before insert.
    """

    __tablename__ = "stg_group_entry"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    iteration_id: Mapped[int] = mapped_column(ForeignKey("stg_group_iteration.id"))
    node_id: Mapped[int]  # FK to index.db node
    cluster_id: Mapped[Optional[int]]
    pre_processed_name: Mapped[Optional[str]]
    processed_name: Mapped[Optional[str]]
    derived_names_json: Mapped[Optional[str]]
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    processed: Mapped[bool] = mapped_column(default=False)

    # Relationship
    iteration: Mapped["GroupIteration"] = relationship(back_populates="entries")

    __table_args__ = (
        Index("idx_group_entry_iter", "iteration_id"),
        Index("idx_group_entry_node", "node_id"),
    )


class GroupCategory(WorkBase):
    """Category/group discovered during grouping iteration."""

    __tablename__ = "stg_group_category"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    iteration_id: Mapped[int] = mapped_column(ForeignKey("stg_group_iteration.id"))
    name: Mapped[str] = mapped_column(String)
    count: Mapped[Optional[int]]
    group_confidence: Mapped[Optional[float]] = mapped_column(Float)
    needs_review: Mapped[bool] = mapped_column(default=False)
    reviewed: Mapped[bool] = mapped_column(default=False)

    # Relationship
    iteration: Mapped["GroupIteration"] = relationship(back_populates="categories")

    __table_args__ = (Index("idx_group_category_iter", "iteration_id"),)


class GroupCategoryEntry(WorkBase):
    """Maps folders to groups through their partial name categories"""

    __tablename__ = "stg_group_category_entries"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    folder_id: Mapped[int]
    partial_category_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("partial_name_categories.id"), index=True
    )
    group_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("stg_group_category.id"), index=True
    )
    iteration_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("stg_group_iteration.id"), index=True
    )
    cluster_id: Mapped[Optional[int]]
    processed_name: Mapped[Optional[str]]
    pre_processed_name: Mapped[Optional[str]]
    derived_names: Mapped[Optional[List]] = mapped_column(JsonList)
    path: Mapped[Optional[str]]
    confidence: Mapped[float] = mapped_column(Float, default=0)
    processed: Mapped[bool] = mapped_column(default=False)

    # Relationships
    # folder: Mapped["Folder"] = relationship(back_populates="group_entries")
    # partial_category: Mapped[Optional["PartialNameCategory"]] = relationship(back_populates="group_entries")
    # group: Mapped[Optional["GroupCategory"]] = relationship(back_populates="entries")

    def __repr__(self):
        return f"GroupCategoryEntry(id={self.id}, original={self.pre_processed_name}, processed={self.processed_name})"


class PartialNameCategory(WorkBase):
    """
    Represents a part of a folder name, once the string has been broken
    down into best-guess categories.
    (i.e. "garden indoor" -> "garden" and "indoor")
    """

    __tablename__ = "partial_name_categories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[Optional[str]] = mapped_column(String, index=True)
    original_name: Mapped[Optional[str]]
    classification: Mapped[Optional[str]]
    node_id: Mapped[int]
    hidden: Mapped[bool] = mapped_column(default=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)

    def __repr__(self):
        return f"PartialNameCategory(id={self.id}, name={self.name})"


class Classification(WorkBase):
    """Classification result for a node (variant/collection/subject/uncertain).

    IMPORTANT: node_id references index.db and must be validated before insert.
    """

    __tablename__ = "stg_classification"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run.id"))
    node_id: Mapped[int]
    classification: Mapped[
        Optional[str]
    ]  # 'variant' | 'collection' | 'subject' | 'uncertain'
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    method: Mapped[Optional[str]]  # 'structural' | 'llm' | 'manual'

    __table_args__ = (
        Index("idx_classification_run", "run_id"),
        Index("idx_classification_node", "node_id"),
        Index("idx_classification_run_node", "run_id", "node_id", unique=True),
    )


class FolderStructure(WorkBase):
    """
    Represents the most-recently calculated folder structure
    for the old and new structures. This should be serializable into
    data_models.api.FolderV2 and data_models.api.File objects via
    pydantic
    """

    __tablename__ = "out_folder_structure"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("run.id"))
    snapshot_id: Mapped[int]  # FK to index.db (cross-database)
    total_nodes: Mapped[int]
    structure_type: Mapped[str] = mapped_column(String)
    structure: Mapped[dict] = mapped_column(JsonDict)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime)


class FileMapping(WorkBase):
    """Mapping from original file path to new organized path.

    IMPORTANT: node_id references index.db and must be validated before insert.
    """

    __tablename__ = "out_file_mapping"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run.id"))
    node_id: Mapped[int]
    original_path: Mapped[str] = mapped_column(String)
    new_path: Mapped[Optional[str]]
    groups: Mapped[Optional[dict]] = mapped_column(JsonDict)

    __table_args__ = (
        Index("idx_file_mapping_run", "run_id"),
        Index("idx_file_mapping_node", "node_id"),
        Index("idx_file_mapping_run_node", "run_id", "node_id", unique=True),
    )


class Meta(WorkBase):
    """Metadata key-value store for work.db.

    Used for storing schema_version and other database-level metadata.
    """

    __tablename__ = "meta"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[Optional[str]]
