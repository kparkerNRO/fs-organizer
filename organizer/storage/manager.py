"""Storage manager for index.db, work.db, and training.db databases.

This module provides the main interface for interacting with the databases:
filesystem index (index.db), intermediary work (work.db), and training (training.db).
Configuration data is managed separately via YAML files loaded in-memory
(see organizer/utils/config.py).
"""

from enum import Enum
from dataclasses import dataclass

from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Iterator
from datetime import datetime

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from utils.config import compute_reference_hash

from storage.index_models import (
    IndexBase,
    Snapshot,
    Node,
    Meta as IndexMeta,
    INDEX_SCHEMA_VERSION,
)
from storage.work_models import (
    WorkBase,
    Run,
    Meta as WorkMeta,
    WORK_SCHEMA_VERSION,
)


# Default paths
DATA_DIR = Path(__file__).parent.parent / "data"
# INDEX_DB = DATA_DIR / "index.db"
# WORK_DB = DATA_DIR /  "work.db"


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


class NodeKind(str, Enum):
    """Allowed values for Node.kind."""

    FILE = "file"
    DIR = "dir"


class FileSource(str, Enum):
    """Allowed values for Node.file_source."""

    FILESYSTEM = "filesystem"
    ZIP_FILE = "zip_file"
    ZIP_CONTENT = "zip_content"


class RunStatus(str, Enum):
    """Allowed values for Run.status."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class IngestionJob:
    """Context payload for an ingestion job."""

    storage: "StorageManager"
    snapshot_id: int
    run_id: int


class StorageManager:
    """Manager for index.db, work.db, and training.db databases.

    This class provides the main interface for creating snapshots, managing runs,
    and querying data across both databases.

    IMPORTANT:
    - Snapshots in index.db are immutable after creation
    - Cross-database references (snapshot_id, node_id) are validated at application level
    - Schema versions are checked on initialization
    """

    def __init__(
        self,
        database_path: Path | None,
        *,
        index_path: Path | None = None,
        work_path: Path | None = None,
        training_path: Path | None = None,
        enable_index: bool = True,
        enable_work: bool = True,
        enable_training: bool = True,
    ):
        """Initialize storage manager.

        Args:
            database_path: Path to output path where the databases live (defaults to data/)
        """
        if database_path is None:
            database_path = DATA_DIR

        self._index_enabled = enable_index
        self._work_enabled = enable_work
        self._training_enabled = enable_training

        self.index_path = index_path or (
            database_path / "index.db" if enable_index else None
        )
        self.work_path = work_path or (
            database_path / "work.db" if enable_work else None
        )
        self.training_path = training_path or (
            database_path / "training.db" if enable_training else None
        )

        # Store engines for later use
        self.index_engine = None
        self.work_engine = None
        self.training_engine = None

        # Ensure databases exist and schemas are initialized
        self._ensure_databases()

    def _ensure_databases(self):
        """Ensure database files exist and schemas are initialized."""
        # Create parent directories
        if self._index_enabled and self.index_path:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_index_schema()

        if self._work_enabled and self.work_path:
            self.work_path.parent.mkdir(parents=True, exist_ok=True)
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

        PRAGMAs are set via event listener (see module-level decorator)
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

    def ensure_training_db(self) -> None:
        """Ensure training.db exists and has expected schema version."""
        if not self._training_enabled or not self.training_path:
            raise RuntimeError(
                "Training database is disabled for this storage manager."
            )

        if self.training_engine is None:
            self._init_training_schema()

    def _init_training_schema(self) -> None:
        """Initialize training.db schema and verify version."""
        from storage.training_models import TRAINING_SCHEMA_VERSION, Meta, TrainingBase

        if str(self.training_path) == ":memory:":
            db_url = "sqlite:///:memory:"
        else:
            self.training_path.parent.mkdir(parents=True, exist_ok=True)
            db_url = f"sqlite:///{self.training_path}"

        self.training_engine = create_engine(db_url)
        TrainingBase.metadata.create_all(self.training_engine)
        self._verify_training_schema_version(
            self.training_engine, TRAINING_SCHEMA_VERSION, Meta
        )

    @staticmethod
    def _verify_training_schema_version(engine, expected_version, meta_model) -> None:
        from sqlalchemy.orm import Session

        session = Session(engine)
        try:
            meta = session.query(meta_model).filter_by(key="schema_version").first()
            if meta is None:
                meta = meta_model(key="schema_version", value=expected_version)
                session.add(meta)
                session.commit()
            elif meta.value != expected_version:
                raise RuntimeError(
                    "training.db schema version mismatch: "
                    f"database is v{meta.value}, code expects v{expected_version}. "
                    "Delete the training.db file to recreate."
                )
        finally:
            session.close()

    @contextmanager
    def get_index_session(self, read_only: bool = False) -> Iterator[Session]:
        """Get SQLAlchemy session for index.db.

        Args:
            read_only: If True, returns session that raises error on flush/commit.
                      Use for snapshot queries to prevent accidental mutations.

        Returns:
            Context manager yielding a SQLAlchemy session

        IMPORTANT: Snapshots are immutable. When querying snapshots, prefer
        read_only=True to prevent accidental modifications via session.
        """
        if not self._index_enabled or self.index_engine is None:
            raise RuntimeError("Index database is disabled for this storage manager.")

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

        try:
            yield session
        finally:
            session.close()

    @contextmanager
    def get_work_session(self) -> Iterator[Session]:
        """Get SQLAlchemy session for work.db.

        Returns:
            Context manager yielding a SQLAlchemy session
        """
        if not self._work_enabled or self.work_engine is None:
            raise RuntimeError("Work database is disabled for this storage manager.")

        Session = sessionmaker(bind=self.work_engine)
        session = Session()
        try:
            yield session
        finally:
            session.close()

    @contextmanager
    def get_training_session(self) -> Iterator[Session]:
        """Get SQLAlchemy session for training.db."""
        self.ensure_training_db()

        Session = sessionmaker(bind=self.training_engine)
        session = Session()
        try:
            yield session
        finally:
            session.close()

    # Referential integrity validation methods

    def _validate_snapshot_exists(self, snapshot_id: int) -> bool:
        """Check if snapshot exists in index.db.

        Args:
            snapshot_id: Snapshot ID to check

        Returns:
            True if snapshot exists, False otherwise
        """
        with self.get_index_session(read_only=True) as session:
            return (
                session.query(Snapshot).filter_by(snapshot_id=snapshot_id).first()
                is not None
            )

    def _validate_node_exists(self, node_id: int, snapshot_id: int) -> bool:
        """Check if node exists in index.db for given snapshot.

        Args:
            node_id: Node ID to check
            snapshot_id: Snapshot ID the node should belong to

        Returns:
            True if node exists in snapshot, False otherwise
        """
        with self.get_index_session(read_only=True) as session:
            return (
                session.query(Node)
                .filter_by(node_id=node_id, snapshot_id=snapshot_id)
                .first()
                is not None
            )

    def _check_snapshot_has_runs(self, snapshot_id: int) -> bool:
        """Check if any runs reference this snapshot in work.db.

        Args:
            snapshot_id: Snapshot ID to check

        Returns:
            True if runs exist that reference snapshot, False otherwise
        """
        with self.get_work_session() as session:
            return (
                session.query(Run).filter_by(snapshot_id=snapshot_id).first()
                is not None
            )

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
        with self.get_work_session() as session:
            run = session.query(Run).filter_by(run_id=run_id).first()
            if not run:
                return False
            return run.snapshot_id == snapshot_id

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
        with self.get_index_session() as session:
            snapshot = (
                session.query(Snapshot).filter_by(snapshot_id=snapshot_id).first()
            )
            if snapshot:
                session.delete(
                    snapshot
                )  # Cascades to nodes, node_features via SQLAlchemy
                session.commit()

    def delete_run(self, run_id: int):
        """Delete run and all associated work data.

        Args:
            run_id: Run ID to delete
        """
        with self.get_work_session() as session:
            run = session.query(Run).filter_by(run_id=run_id).first()
            if run:
                # SQLAlchemy cascade deletes stages, group_iterations, etc.
                session.delete(run)
                session.commit()

    def _finish_run(self, run_id: int, status: RunStatus) -> None:
        """Mark a run as finished with a status and timestamp."""
        finished_at = datetime.utcnow().isoformat()
        with self.get_work_session() as session:
            run = session.query(Run).filter_by(run_id=run_id).first()
            if run:
                run.status = status.value
                run.finished_at = finished_at
                session.commit()

    @contextmanager
    def ingestion_job(
        self,
        root_path: Path,
        preprocess_version: Optional[str] = None,
        preprocess_hash: Optional[str] = None,
        pipeline_version: Optional[str] = None,
        config_hash: Optional[str] = None,
        model_id: Optional[str] = None,
        notes: Optional[str] = None,
        reference_hash: Optional[str] = None,
    ) -> Iterator[IngestionJob]:
        """Create a snapshot + run for a new ingestion job.

        The ingestion code can use the returned StorageManager to add nodes,
        update stages, or write work data while the job is running.
        """
        created_at = datetime.utcnow().isoformat()
        root_path_value = Path(root_path)
        reference_hash_value = reference_hash or compute_reference_hash()
        config_hash_value = config_hash or reference_hash_value

        with self.get_index_session() as index_session:
            snapshot = Snapshot(
                created_at=created_at,
                root_path=str(root_path_value),
                root_abs_path=str(root_path_value.resolve()),
                preprocess_version=preprocess_version,
                preprocess_hash=preprocess_hash,
                reference_hash=reference_hash_value,
                notes=notes,
            )
            index_session.add(snapshot)
            index_session.commit()
            snapshot_id = snapshot.snapshot_id

        with self.get_work_session() as work_session:
            run = Run(
                snapshot_id=snapshot_id,
                started_at=created_at,
                status=RunStatus.RUNNING.value,
                pipeline_version=pipeline_version,
                config_hash=config_hash_value,
                model_id=model_id,
                notes=notes,
            )
            work_session.add(run)
            work_session.commit()
            run_id = run.run_id

        job = IngestionJob(storage=self, snapshot_id=snapshot_id, run_id=run_id)

        try:
            yield job
        except Exception:
            self._finish_run(run_id, RunStatus.FAILED)
            raise
        else:
            self._finish_run(run_id, RunStatus.COMPLETED)

    # Placeholder methods for modification prevention

    def update_node(self, node_id: int, **kwargs):
        """NOT ALLOWED: Nodes cannot be modified after snapshot creation.

        Raises:
            NotImplementedError: Always, as nodes are immutable
        """
        raise NotImplementedError(
            "Nodes are immutable. Create a new snapshot to capture changes."
        )

    def compute_node_features(self, snapshot_id: int):
        """NOT ALLOWED: Features must be computed during snapshot creation.

        Raises:
            NotImplementedError: Always, as features must be computed atomically
        """
        raise NotImplementedError(
            "Node features are computed during ingest_filesystem(). "
            "This method should not be called externally."
        )
