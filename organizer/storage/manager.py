"""Storage manager for index.db and work.db databases.

This module provides the main interface for interacting with the two-database
architecture: filesystem index (index.db) and intermediary work (work.db).
Configuration data is managed separately via YAML files loaded in-memory
(see organizer/utils/config.py).
"""

from enum import Enum
from dataclasses import dataclass

from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Iterator
from datetime import datetime, UTC

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
from storage.training_models import (
    TrainingBase,
    Meta as TrainingMeta,
    TRAINING_SCHEMA_VERSION,
)


# Default paths
DATA_DIR = Path(__file__).parent.parent / "data"


# CRITICAL: Set PRAGMAs per connection, not per engine
# SQLite requires these settings on every new connection
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Set SQLite PRAGMAs for every new connection."""
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
    """Manager for index.db, work.db, and training.db databases."""

    def __init__(self, storage_path: Path | None):
        """Initialize storage manager for all databases."""
        if storage_path is None:
            storage_path = DATA_DIR
        self.storage_path = storage_path
        self.index_path = storage_path / "index.db"
        self.work_path = storage_path / "work.db"
        self.training_path = storage_path / "training.db"

        self.index_engine: Optional[Engine] = None
        self.work_engine: Optional[Engine] = None
        self.training_engine: Optional[Engine] = None

        self._ensure_databases()

    def get_index_db_path(self) -> Path:
        return self.index_path

    def get_work_db_path(self) -> Path:
        return self.work_path

    def get_training_db_path(self) -> Path:
        return self.training_path

    def _ensure_databases(self):
        """Ensure database files exist and schemas are initialized."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._init_index_schema()
        self._init_work_schema()
        self._init_training_schema()

    def _init_schema(
        self, db_path: Path, base, version_key: str, expected_version: str, meta_model
    ) -> Engine:
        """Generic schema initializer for a database."""
        engine = create_engine(f"sqlite:///{db_path}")
        base.metadata.create_all(engine)

        with Session(engine) as session:
            meta = session.query(meta_model).filter_by(key=version_key).first()
            if meta is None:
                meta = meta_model(key=version_key, value=expected_version)
                session.add(meta)
                session.commit()
            elif meta.value != expected_version:
                raise RuntimeError(
                    f"{db_path.name} schema version mismatch: "
                    f"database is v{meta.value}, code expects v{expected_version}."
                )
        return engine

    def _init_index_schema(self):
        """Initialize index.db schema and verify version."""
        self.index_engine = self._init_schema(
            self.index_path,
            IndexBase,
            "schema_version",
            INDEX_SCHEMA_VERSION,
            IndexMeta,
        )

    def _init_work_schema(self):
        """Initialize work.db schema and verify version."""
        self.work_engine = self._init_schema(
            self.work_path, WorkBase, "schema_version", WORK_SCHEMA_VERSION, WorkMeta
        )

    def _init_training_schema(self):
        """Initialize training.db schema and verify version."""
        self.training_engine = self._init_schema(
            self.training_path,
            TrainingBase,
            "schema_version",
            TRAINING_SCHEMA_VERSION,
            TrainingMeta,
        )

    @contextmanager
    def get_session(self, db_type: str, read_only: bool = False) -> Iterator[Session]:
        """Get a SQLAlchemy session for the specified database type."""
        if db_type == "index":
            engine = self.index_engine
        elif db_type == "work":
            engine = self.work_engine
        elif db_type == "training":
            engine = self.training_engine
        else:
            raise ValueError(f"Unknown database type: {db_type}")

        if engine is None:
            raise RuntimeError(f"{db_type} engine not initialized.")

        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()

        if read_only:

            @event.listens_for(session, "before_flush")
            def prevent_flush(session, flush_context, instances):
                raise RuntimeError(
                    f"Cannot modify {db_type}.db with read-only session."
                )

        try:
            yield session
        finally:
            session.close()

    def get_index_session(self, read_only: bool = False) -> Iterator[Session]:
        """Context manager for an index.db session."""
        return self.get_session("index", read_only=read_only)

    def get_work_session(self) -> Iterator[Session]:
        """Context manager for a work.db session."""
        return self.get_session("work")

    def get_training_session(self) -> Iterator[Session]:
        """Context manager for a training.db session."""
        return self.get_session("training")

    # ... (rest of the class remains the same)
    # ...

    def _validate_snapshot_exists(self, snapshot_id: int) -> bool:
        """Check if snapshot exists in index.db."""
        with self.get_index_session(read_only=True) as session:
            return (
                session.query(Snapshot).filter_by(snapshot_id=snapshot_id).first()
                is not None
            )

    def _validate_node_exists(self, node_id: int, snapshot_id: int) -> bool:
        """Check if node exists in index.db for given snapshot."""
        with self.get_index_session(read_only=True) as session:
            return (
                session.query(Node)
                .filter_by(node_id=node_id, snapshot_id=snapshot_id)
                .first()
                is not None
            )

    def _check_snapshot_has_runs(self, snapshot_id: int) -> bool:
        """Check if any runs reference this snapshot in work.db."""
        with self.get_work_session() as session:
            return (
                session.query(Run).filter_by(snapshot_id=snapshot_id).first()
                is not None
            )

    def _validate_snapshot_id_matches_run(self, snapshot_id: int, run_id: int) -> bool:
        """Validate that provided snapshot_id matches the run's snapshot_id."""
        with self.get_work_session() as session:
            run = session.query(Run).filter_by(run_id=run_id).first()
            if not run:
                return False
            return run.snapshot_id == snapshot_id

    def delete_snapshot(self, snapshot_id: int):
        """Delete snapshot. Fails if runs exist that reference it."""
        if self._check_snapshot_has_runs(snapshot_id):
            raise ValueError(
                f"Cannot delete snapshot {snapshot_id}: "
                f"runs still reference it. Delete runs first."
            )
        with self.get_index_session() as session:
            snapshot = (
                session.query(Snapshot).filter_by(snapshot_id=snapshot_id).first()
            )
            if snapshot:
                session.delete(snapshot)
                session.commit()

    def delete_run(self, run_id: int):
        """Delete run and all associated work data."""
        with self.get_work_session() as session:
            run = session.query(Run).filter_by(run_id=run_id).first()
            if run:
                session.delete(run)
                session.commit()

    def _finish_run(self, run_id: int, status: RunStatus) -> None:
        """Mark a run as finished with a status and timestamp."""
        finished_at = datetime.now(UTC).isoformat()
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
        """Create a snapshot + run for a new ingestion job."""
        created_at = datetime.now(UTC).isoformat()
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

    def update_node(self, node_id: int, **kwargs):
        """NOT ALLOWED: Nodes cannot be modified after snapshot creation."""
        raise NotImplementedError(
            "Nodes are immutable. Create a new snapshot to capture changes."
        )

    def compute_node_features(self, snapshot_id: int):
        """NOT ALLOWED: Features must be computed during snapshot creation."""
        raise NotImplementedError(
            "Node features are computed during ingest_filesystem(). "
            "This method should not be called externally."
        )
