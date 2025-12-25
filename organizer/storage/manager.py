"""Storage manager for index.db and work.db databases.

This module provides the main interface for interacting with the two-database
architecture: filesystem index (index.db) and intermediary work (work.db).
Configuration data is managed separately via YAML files loaded in-memory
(see organizer/utils/config.py).
"""

from pathlib import Path
from typing import Optional
from datetime import datetime

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from storage.index_models import (
    IndexBase,
    Snapshot,
    Node,
    NodeFeatures,
    Meta as IndexMeta,
    INDEX_SCHEMA_VERSION,
)
from storage.work_models import (
    WorkBase,
    Run,
    StageState,
    GroupIteration,
    Meta as WorkMeta,
    WORK_SCHEMA_VERSION,
)

# Default paths
DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_DB = DATA_DIR / "index" / "index.db"
WORK_DB = DATA_DIR / "work" / "work.db"


# CRITICAL: Set PRAGMAs per connection, not per engine
# SQLite requires these settings on every new connection
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Set SQLite PRAGMAs for every new connection.

    CRITICAL: This must be done per connection, not just once during engine init.
    Without this, new connections will silently disable foreign key enforcement.
    """
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


class StorageManager:
    """Manager for index.db and work.db databases.

    This class provides the main interface for creating snapshots, managing runs,
    and querying data across both databases.

    IMPORTANT:
    - Snapshots in index.db are immutable after creation
    - Cross-database references (snapshot_id, node_id) are validated at application level
    - Schema versions are checked on initialization
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        work_path: Optional[Path] = None,
    ):
        """Initialize storage manager.

        Args:
            index_path: Path to index.db (defaults to data/index/index.db)
            work_path: Path to work.db (defaults to data/work/work.db)
        """
        self.index_path = index_path or INDEX_DB
        self.work_path = work_path or WORK_DB

        # Store engines for later use
        self.index_engine = None
        self.work_engine = None

        # Ensure databases exist and schemas are initialized
        self._ensure_databases()

    def _ensure_databases(self):
        """Ensure database files exist and schemas are initialized."""
        # Create parent directories
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.work_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schemas
        self._init_index_schema()
        self._init_work_schema()

    def _init_index_schema(self):
        """Initialize index.db schema and verify version.

        CRITICAL: PRAGMAs are set via event listener (see module-level decorator)
        to ensure they apply to ALL connections, not just the first one.
        """
        # Handle in-memory database path
        if str(self.index_path) == ":memory:":
            db_url = "sqlite:///:memory:"
        else:
            db_url = f"sqlite:///{self.index_path}"

        self.index_engine = create_engine(db_url)

        # Create tables if needed
        IndexBase.metadata.create_all(self.index_engine)

        # Check/set schema version
        self._verify_index_schema_version(self.index_engine)

    def _verify_index_schema_version(self, engine):
        """Verify index.db schema version matches code version.

        Args:
            engine: SQLAlchemy engine for index.db

        Raises:
            RuntimeError: If schema version mismatch detected
        """
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            # Get stored version
            meta = session.query(IndexMeta).filter_by(key="schema_version").first()

            if meta is None:
                # New database, set version
                meta = IndexMeta(key="schema_version", value=INDEX_SCHEMA_VERSION)
                session.add(meta)
                session.commit()
            else:
                # Existing database, check version
                if meta.value != INDEX_SCHEMA_VERSION:
                    raise RuntimeError(
                        f"index.db schema version mismatch: "
                        f"database is v{meta.value}, code expects v{INDEX_SCHEMA_VERSION}. "
                        f"Delete {self.index_path} and re-run gather to regenerate snapshots."
                    )
        finally:
            session.close()

    def _init_work_schema(self):
        """Initialize work.db schema and verify version.

        CRITICAL: PRAGMAs are set via event listener (see module-level decorator)
        to ensure they apply to ALL connections, not just the first one.
        """
        # Handle in-memory database path
        if str(self.work_path) == ":memory:":
            db_url = "sqlite:///:memory:"
        else:
            db_url = f"sqlite:///{self.work_path}"

        self.work_engine = create_engine(db_url)

        # Create tables if needed
        WorkBase.metadata.create_all(self.work_engine)

        # Check/set schema version
        self._verify_work_schema_version(self.work_engine)

    def _verify_work_schema_version(self, engine):
        """Verify work.db schema version matches code version.

        Args:
            engine: SQLAlchemy engine for work.db

        Raises:
            RuntimeError: If schema version mismatch detected
        """
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            # Get stored version
            meta = session.query(WorkMeta).filter_by(key="schema_version").first()

            if meta is None:
                # New database, set version
                meta = WorkMeta(key="schema_version", value=WORK_SCHEMA_VERSION)
                session.add(meta)
                session.commit()
            else:
                # Existing database, check version
                if meta.value != WORK_SCHEMA_VERSION:
                    raise RuntimeError(
                        f"work.db schema version mismatch: "
                        f"database is v{meta.value}, code expects v{WORK_SCHEMA_VERSION}. "
                        f"Delete {self.work_path} to recreate (WARNING: loses all run data)."
                    )
        finally:
            session.close()

    def get_index_session(self, read_only: bool = False) -> Session:
        """Get SQLAlchemy session for index.db.

        Args:
            read_only: If True, returns session that raises error on flush/commit.
                      Use for snapshot queries to prevent accidental mutations.

        Returns:
            SQLAlchemy session

        IMPORTANT: Snapshots are immutable. When querying snapshots, prefer
        read_only=True to prevent accidental modifications via session.
        """
        Session = sessionmaker(bind=self.index_engine)
        session = Session()

        if read_only:
            # Prevent writes by raising on flush
            @event.listens_for(session, "before_flush")
            def prevent_flush(session, flush_context, instances):
                raise RuntimeError(
                    "Cannot modify index.db with read-only session. "
                    "Snapshots are immutable after creation."
                )

        return session

    def get_work_session(self) -> Session:
        """Get SQLAlchemy session for work.db.

        Returns:
            SQLAlchemy session
        """
        Session = sessionmaker(bind=self.work_engine)
        return Session()

    # Referential integrity validation methods

    def _validate_snapshot_exists(self, snapshot_id: int) -> bool:
        """Check if snapshot exists in index.db.

        Args:
            snapshot_id: Snapshot ID to check

        Returns:
            True if snapshot exists, False otherwise
        """
        session = self.get_index_session(read_only=True)
        try:
            return session.query(Snapshot).filter_by(snapshot_id=snapshot_id).first() is not None
        finally:
            session.close()

    def _validate_node_exists(self, node_id: int, snapshot_id: int) -> bool:
        """Check if node exists in index.db for given snapshot.

        Args:
            node_id: Node ID to check
            snapshot_id: Snapshot ID the node should belong to

        Returns:
            True if node exists in snapshot, False otherwise
        """
        session = self.get_index_session(read_only=True)
        try:
            return (
                session.query(Node)
                .filter_by(node_id=node_id, snapshot_id=snapshot_id)
                .first()
                is not None
            )
        finally:
            session.close()

    def _check_snapshot_has_runs(self, snapshot_id: int) -> bool:
        """Check if any runs reference this snapshot in work.db.

        Args:
            snapshot_id: Snapshot ID to check

        Returns:
            True if runs exist that reference snapshot, False otherwise
        """
        session = self.get_work_session()
        try:
            return session.query(Run).filter_by(snapshot_id=snapshot_id).first() is not None
        finally:
            session.close()

    def _validate_snapshot_id_matches_run(self, snapshot_id: int, run_id: int) -> bool:
        """Validate that provided snapshot_id matches the run's snapshot_id.

        CRITICAL: Work tables (StageState, GroupIteration) store snapshot_id redundantly.
        This method prevents inconsistencies where snapshot_id diverges from run.snapshot_id.

        Args:
            snapshot_id: Snapshot ID to validate
            run_id: Run ID to check against

        Returns:
            True if snapshot_id matches run's snapshot_id, False otherwise
        """
        session = self.get_work_session()
        try:
            run = session.query(Run).filter_by(run_id=run_id).first()
            if not run:
                return False
            return run.snapshot_id == snapshot_id
        finally:
            session.close()

    # Deletion methods with referential integrity

    def delete_snapshot(self, snapshot_id: int):
        """Delete snapshot. Fails if runs exist that reference it.

        Args:
            snapshot_id: Snapshot ID to delete

        Raises:
            ValueError: If runs still reference this snapshot
        """
        # CRITICAL: Check for referencing runs
        if self._check_snapshot_has_runs(snapshot_id):
            raise ValueError(
                f"Cannot delete snapshot {snapshot_id}: "
                f"runs still reference it. Delete runs first."
            )

        # Safe to delete
        session = self.get_index_session()
        try:
            snapshot = session.query(Snapshot).filter_by(snapshot_id=snapshot_id).first()
            if snapshot:
                session.delete(snapshot)  # Cascades to nodes, node_features via SQLAlchemy
                session.commit()
        finally:
            session.close()

    def delete_run(self, run_id: int):
        """Delete run and all associated work data.

        Args:
            run_id: Run ID to delete
        """
        session = self.get_work_session()
        try:
            run = session.query(Run).filter_by(run_id=run_id).first()
            if run:
                # SQLAlchemy cascade deletes stages, group_iterations, etc.
                session.delete(run)
                session.commit()
        finally:
            session.close()

    # Placeholder methods for modification prevention

    def update_node(self, node_id: int, **kwargs):
        """NOT ALLOWED: Nodes cannot be modified after snapshot creation.

        Raises:
            NotImplementedError: Always, as nodes are immutable
        """
        raise NotImplementedError("Nodes are immutable. Create a new snapshot to capture changes.")

    def compute_node_features(self, snapshot_id: int):
        """NOT ALLOWED: Features must be computed during snapshot creation.

        Raises:
            NotImplementedError: Always, as features must be computed atomically
        """
        raise NotImplementedError(
            "Node features are computed during ingest_filesystem(). "
            "This method should not be called externally."
        )
