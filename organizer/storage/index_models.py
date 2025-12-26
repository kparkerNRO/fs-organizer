"""Index database models for filesystem snapshots.

This module defines SQLAlchemy models for the index.db database, which stores
immutable snapshots of filesystem state.

IMPORTANT: Snapshots are write-once, read-many. Once created via
ingest_filesystem(), they MUST NOT be modified. All nodes and node_features
are computed atomically during creation.
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    ForeignKey,
    CheckConstraint,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship

# Schema version (increment on breaking changes)
INDEX_SCHEMA_VERSION = "1.0.0"

IndexBase = declarative_base()


class Snapshot(IndexBase):
    """Immutable snapshot of filesystem state.

    IMPORTANT: Snapshots are write-once, read-many. Once created via
    ingest_filesystem(), they MUST NOT be modified. All nodes and node_features
    are computed atomically during creation.

    To capture a new state of the filesystem, create a new snapshot instead
    of modifying an existing one.
    """

    __tablename__ = "snapshot"

    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(String, nullable=False)
    root_path = Column(String, nullable=False)
    root_abs_path = Column(String, nullable=False)
    preprocess_version = Column(String)
    preprocess_hash = Column(String)
    reference_hash = Column(String)
    notes = Column(String)

    # Relationships
    nodes = relationship(
        "Node", back_populates="snapshot", cascade="all, delete-orphan"
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

    node_id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(Integer, ForeignKey("snapshot.snapshot_id"), nullable=False)
    parent_node_id = Column(Integer, ForeignKey("node.node_id"), nullable=True)
    kind = Column(String, CheckConstraint("kind IN ('file', 'dir')"), nullable=False)
    name = Column(String, nullable=False)
    rel_path = Column(String, nullable=False)
    abs_path = Column(String, nullable=False)
    ext = Column(String)
    size = Column(Integer)
    mtime = Column(Float)
    ctime = Column(Float)
    inode = Column(Integer)
    depth = Column(Integer, nullable=False)

    # CRITICAL: file_source must be NOT NULL to ensure unique constraint works
    # Values: 'filesystem' | 'zip_file' | 'zip_content'
    file_source = Column(String, nullable=False, default="filesystem")

    num_folder_children = Column(Integer, default=0)
    num_file_children = Column(Integer, default=0)

    # Relationships
    snapshot = relationship("Snapshot", back_populates="nodes")
    parent = relationship("Node", remote_side=[node_id], backref="children")
    features = relationship(
        "NodeFeatures",
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

    node_id = Column(Integer, ForeignKey("node.node_id"), primary_key=True)
    normalized_name = Column(String)
    tokens_json = Column(String)  # JSON array
    hints_json = Column(String)  # JSON object

    # Relationship
    node = relationship("Node", back_populates="features")


class Meta(IndexBase):
    """Metadata key-value store for index.db.

    Used for storing schema_version and other database-level metadata.
    """

    __tablename__ = "meta"

    key = Column(String, primary_key=True)
    value = Column(String)
