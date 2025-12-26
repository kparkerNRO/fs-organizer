"""Index database models for filesystem snapshots.

This module defines SQLAlchemy models for the index.db database, which stores
immutable snapshots of filesystem state.

IMPORTANT: Snapshots are write-once, read-many. Once created via
ingest_filesystem(), they MUST NOT be modified. All nodes and node_features
are computed atomically during creation.
"""

from typing import List, Optional
from sqlalchemy import (
    String,
    Integer,
    Float,
    ForeignKey,
    CheckConstraint,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column

# Schema version (increment on breaking changes)
INDEX_SCHEMA_VERSION = "1.0.0"


class IndexBase(DeclarativeBase):
    pass


class Snapshot(IndexBase):
    """Immutable snapshot of filesystem state.

    IMPORTANT: Snapshots are write-once, read-many. Once created via
    ingest_filesystem(), they MUST NOT be modified. All nodes and node_features
    are computed atomically during creation.

    To capture a new state of the filesystem, create a new snapshot instead
    of modifying an existing one.
    """

    __tablename__ = "snapshot"

    snapshot_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    root_path: Mapped[str] = mapped_column(String, nullable=False)
    root_abs_path: Mapped[str] = mapped_column(String, nullable=False)
    preprocess_version: Mapped[Optional[str]] = mapped_column(String)
    preprocess_hash: Mapped[Optional[str]] = mapped_column(String)
    reference_hash: Mapped[Optional[str]] = mapped_column(String)
    notes: Mapped[Optional[str]] = mapped_column(String)

    # Relationships
    nodes: Mapped[List["Node"]] = relationship(
        back_populates="snapshot", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("idx_snapshot_root", "root_abs_path"),)


class Node(IndexBase):
    """Filesystem node (file or directory) within a snapshot.

    CRITICAL Path Namespace:
    - file_source distinguishes between filesystem paths and ZIP content paths
    - Unique constraint on (snapshot_id, rel_path, file_source) prevents collisions
    - Example: real directory "a.zip/" and ZIP file "a.zip" can coexist

    CRITICAL ZIP Parent Modeling:
    - ZIP files (file_source='zip_file', kind='file') act as virtual directories
    - ZIP content nodes have parent_node_id pointing to the ZIP file node
    - Traversal code must treat file_source='zip_file' nodes as containers
    """

    __tablename__ = "node"

    node_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int] = mapped_column(ForeignKey("snapshot.snapshot_id"), nullable=False)
    parent_node_id: Mapped[Optional[int]] = mapped_column(ForeignKey("node.node_id"), nullable=True)
    kind: Mapped[str] = mapped_column(String, CheckConstraint("kind IN ('file', 'dir')"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    rel_path: Mapped[str] = mapped_column(String, nullable=False)
    abs_path: Mapped[str] = mapped_column(String, nullable=False)
    ext: Mapped[Optional[str]] = mapped_column(String)
    size: Mapped[Optional[int]] = mapped_column(Integer)
    mtime: Mapped[Optional[float]] = mapped_column(Float)
    ctime: Mapped[Optional[float]] = mapped_column(Float)
    inode: Mapped[Optional[int]] = mapped_column(Integer)
    depth: Mapped[int] = mapped_column(Integer, nullable=False)

    # CRITICAL: file_source must be NOT NULL to ensure unique constraint works
    # Values: 'filesystem' | 'zip_file' | 'zip_content'
    file_source: Mapped[str] = mapped_column(String, nullable=False, default="filesystem")

    num_folder_children: Mapped[int] = mapped_column(Integer, default=0)
    num_file_children: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    snapshot: Mapped["Snapshot"] = relationship(back_populates="nodes")
    parent: Mapped[Optional["Node"]] = relationship(remote_side=[node_id], backref="children")
    features: Mapped[Optional["NodeFeatures"]] = relationship(
        back_populates="node",
        uselist=False,
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_node_snapshot", "snapshot_id"),
        Index("idx_node_parent", "snapshot_id", "parent_node_id"),
        Index("idx_node_kind", "snapshot_id", "kind"),
        # CRITICAL: Composite unique key prevents path collisions between
        # filesystem and ZIP content
        Index("idx_node_path", "snapshot_id", "rel_path", "file_source", unique=True),
    )


class NodeFeatures(IndexBase):
    """Computed features for a node (normalized names, tokens, hints).

    Features are computed atomically during snapshot creation and cannot be
    modified afterward.
    """

    __tablename__ = "node_features"

    node_id: Mapped[int] = mapped_column(ForeignKey("node.node_id"), primary_key=True)
    normalized_name: Mapped[Optional[str]] = mapped_column(String)
    tokens_json: Mapped[Optional[str]] = mapped_column(String)  # JSON array
    hints_json: Mapped[Optional[str]] = mapped_column(String)  # JSON object

    # Relationship
    node: Mapped["Node"] = relationship(back_populates="features")


class Meta(IndexBase):
    """Metadata key-value store for index.db.

    Used for storing schema_version and other database-level metadata.
    """

    __tablename__ = "meta"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[Optional[str]] = mapped_column(String)
