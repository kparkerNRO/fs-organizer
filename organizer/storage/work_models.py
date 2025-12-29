"""Work database models for pipeline processing.

This module defines SQLAlchemy models for the work.db database, which stores
intermediary pipeline processing results keyed by snapshot_id + run_id.

IMPORTANT: Cross-database references (snapshot_id, node_id) to index.db are
validated at the application level, not by database constraints.
"""

from typing import List, Optional
from sqlalchemy import String, Float, ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column

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

    run_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int]  # FK to index.db (cross-database)
    started_at: Mapped[str] = mapped_column(String)
    finished_at: Mapped[Optional[str]]
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
    run_id: Mapped[int] = mapped_column(ForeignKey("run.run_id"), primary_key=True)
    completed_at: Mapped[Optional[str]]
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

    iteration_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run.run_id"))
    snapshot_id: Mapped[int]  # Redundant with run.snapshot_id for queries
    timestamp: Mapped[Optional[str]]
    description: Mapped[Optional[str]]
    parameters_json: Mapped[Optional[str]]

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

    entry_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    iteration_id: Mapped[int] = mapped_column(
        ForeignKey("stg_group_iteration.iteration_id")
    )
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

    group_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    iteration_id: Mapped[int] = mapped_column(
        ForeignKey("stg_group_iteration.iteration_id")
    )
    name: Mapped[str] = mapped_column(String)
    count: Mapped[Optional[int]]
    group_confidence: Mapped[Optional[float]] = mapped_column(Float)
    needs_review: Mapped[bool] = mapped_column(default=False)
    reviewed: Mapped[bool] = mapped_column(default=False)

    # Relationship
    iteration: Mapped["GroupIteration"] = relationship(back_populates="categories")

    __table_args__ = (Index("idx_group_category_iter", "iteration_id"),)


class Classification(WorkBase):
    """Classification result for a node (variant/collection/subject/uncertain).

    IMPORTANT: node_id references index.db and must be validated before insert.
    """

    __tablename__ = "stg_classification"

    classification_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run.run_id"))
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
    """Output folder structure generated for a run."""

    __tablename__ = "out_folder_structure"

    structure_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run.run_id"))
    structure_type: Mapped[str] = mapped_column(
        String
    )  # 'original' | 'organized' | 'grouped'
    structure_json: Mapped[str] = mapped_column(String)
    created_at: Mapped[Optional[str]]


class FileMapping(WorkBase):
    """Mapping from original file path to new organized path.

    IMPORTANT: node_id references index.db and must be validated before insert.
    """

    __tablename__ = "out_file_mapping"

    mapping_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run.run_id"))
    node_id: Mapped[int]
    original_path: Mapped[str] = mapped_column(String)
    new_path: Mapped[Optional[str]]
    groups_json: Mapped[Optional[str]]

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
