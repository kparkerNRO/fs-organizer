"""Work database models for pipeline processing.

This module defines SQLAlchemy models for the work.db database, which stores
intermediary pipeline processing results keyed by snapshot_id + run_id.

IMPORTANT: Cross-database references (snapshot_id, node_id) to index.db are
validated at the application level, not by database constraints.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, Index
from sqlalchemy.orm import declarative_base, relationship

# Schema version (increment on breaking changes)
WORK_SCHEMA_VERSION = "1.0.0"

WorkBase = declarative_base()


class Run(WorkBase):
    """Processing run for a specific snapshot.

    Each run represents one execution of the pipeline on a snapshot.
    Multiple runs can be created for the same snapshot to experiment with
    different parameters or pipeline versions.
    """

    __tablename__ = "run"

    run_id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(Integer, nullable=False)  # FK to index.db (cross-database)
    started_at = Column(String, nullable=False)
    finished_at = Column(String)
    status = Column(
        String, default="running"
    )  # 'running' | 'completed' | 'failed' | 'cancelled'
    pipeline_version = Column(String)
    config_hash = Column(String)
    model_id = Column(String)
    notes = Column(String)

    # Relationships
    stages = relationship(
        "StageState", back_populates="run", cascade="all, delete-orphan"
    )
    group_iterations = relationship(
        "GroupIteration", back_populates="run", cascade="all, delete-orphan"
    )


class StageState(WorkBase):
    """Tracks completion state of pipeline stages.

    IMPORTANT: snapshot_id must be validated against run.snapshot_id in
    mark_stage_complete() to prevent inconsistencies.
    """

    __tablename__ = "stage_state"

    stage_name = Column(String, primary_key=True)
    snapshot_id = Column(
        Integer, primary_key=True
    )  # Redundant with run.snapshot_id for query performance
    run_id = Column(Integer, ForeignKey("run.run_id"), primary_key=True)
    completed_at = Column(String)
    input_fingerprint = Column(String)
    output_fingerprint = Column(String)

    # Relationship
    run = relationship("Run", back_populates="stages")


class GroupIteration(WorkBase):
    """Grouping iteration within a run.

    Supports multiple grouping iterations on the same snapshot/run for
    iterative refinement or experimentation.

    IMPORTANT: snapshot_id must be validated against run.snapshot_id in
    create_group_iteration() to prevent inconsistencies.
    """

    __tablename__ = "stg_group_iteration"

    iteration_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("run.run_id"), nullable=False)
    snapshot_id = Column(
        Integer, nullable=False
    )  # Redundant with run.snapshot_id for queries
    timestamp = Column(String)
    description = Column(String)
    parameters_json = Column(String)

    # Relationships
    run = relationship("Run", back_populates="group_iterations")
    entries = relationship(
        "GroupEntry", back_populates="iteration", cascade="all, delete-orphan"
    )
    categories = relationship(
        "GroupCategory", back_populates="iteration", cascade="all, delete-orphan"
    )


class GroupEntry(WorkBase):
    """Entry for a node in a grouping iteration.

    IMPORTANT: node_id references index.db and must be validated before insert.
    """

    __tablename__ = "stg_group_entry"

    entry_id = Column(Integer, primary_key=True, autoincrement=True)
    iteration_id = Column(
        Integer, ForeignKey("stg_group_iteration.iteration_id"), nullable=False
    )
    node_id = Column(Integer, nullable=False)  # FK to index.db node
    cluster_id = Column(Integer)
    pre_processed_name = Column(String)
    processed_name = Column(String)
    derived_names_json = Column(String)
    confidence = Column(Float, default=1.0)
    processed = Column(Boolean, default=False)

    # Relationship
    iteration = relationship("GroupIteration", back_populates="entries")

    __table_args__ = (
        Index("idx_group_entry_iter", "iteration_id"),
        Index("idx_group_entry_node", "node_id"),
    )


class GroupCategory(WorkBase):
    """Category/group discovered during grouping iteration."""

    __tablename__ = "stg_group_category"

    group_id = Column(Integer, primary_key=True, autoincrement=True)
    iteration_id = Column(
        Integer, ForeignKey("stg_group_iteration.iteration_id"), nullable=False
    )
    name = Column(String, nullable=False)
    count = Column(Integer)
    group_confidence = Column(Float)
    needs_review = Column(Boolean, default=False)
    reviewed = Column(Boolean, default=False)

    # Relationship
    iteration = relationship("GroupIteration", back_populates="categories")

    __table_args__ = (Index("idx_group_category_iter", "iteration_id"),)


class Classification(WorkBase):
    """Classification result for a node (variant/collection/subject/uncertain).

    IMPORTANT: node_id references index.db and must be validated before insert.
    """

    __tablename__ = "stg_classification"

    classification_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("run.run_id"), nullable=False)
    node_id = Column(Integer, nullable=False)
    classification = Column(
        String
    )  # 'variant' | 'collection' | 'subject' | 'uncertain'
    confidence = Column(Float)
    method = Column(String)  # 'structural' | 'llm' | 'manual'

    __table_args__ = (
        Index("idx_classification_run", "run_id"),
        Index("idx_classification_node", "node_id"),
        Index("idx_classification_run_node", "run_id", "node_id", unique=True),
    )


class FolderStructure(WorkBase):
    """Output folder structure generated for a run."""

    __tablename__ = "out_folder_structure"

    structure_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("run.run_id"), nullable=False)
    structure_type = Column(
        String, nullable=False
    )  # 'original' | 'organized' | 'grouped'
    structure_json = Column(String, nullable=False)
    created_at = Column(String)


class FileMapping(WorkBase):
    """Mapping from original file path to new organized path.

    IMPORTANT: node_id references index.db and must be validated before insert.
    """

    __tablename__ = "out_file_mapping"

    mapping_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("run.run_id"), nullable=False)
    node_id = Column(Integer, nullable=False)
    original_path = Column(String, nullable=False)
    new_path = Column(String)
    groups_json = Column(String)

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

    key = Column(String, primary_key=True)
    value = Column(String)
