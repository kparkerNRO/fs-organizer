"""Unit tests for StorageManager."""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import cast

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from storage.factories import NodeFactory, RunFactory, SnapshotFactory
from storage.index_models import (
    INDEX_SCHEMA_VERSION,
    IndexBase,
    Snapshot,
)
from storage.index_models import (
    Meta as IndexMeta,
)
from storage.manager import NodeKind, StorageManager
from storage.work_models import (
    WORK_SCHEMA_VERSION,
    Run,
    WorkBase,
)
from storage.work_models import (
    Meta as WorkMeta,
)


@pytest.fixture
def storage_manager(tmp_path: Path) -> StorageManager:
    return StorageManager(database_path=tmp_path)


@pytest.fixture
def index_session(storage_manager: StorageManager):
    with storage_manager.get_index_session() as session:
        NodeFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        SnapshotFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        yield session


@pytest.fixture
def work_session(storage_manager: StorageManager):
    with storage_manager.get_work_session() as session:
        RunFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        yield session


@pytest.fixture
def snapshot(index_session):
    new_snapshot = SnapshotFactory()
    return new_snapshot


@pytest.fixture
def node(index_session, snapshot):
    new_node = NodeFactory(
        id=snapshot.id,
        snapshot_id=snapshot.id,
        kind=NodeKind.FILE,
        name="test.txt",
        rel_path="test.txt",
        abs_path="/test/test.txt",
        depth=1,
    )
    return new_node


@pytest.fixture
def run(work_session, snapshot):
    new_run = cast(
        Run,
        RunFactory(
            id=snapshot.id,
        ),
    )
    return new_run


class TestStorageManager:
    """Test StorageManager initialization and database creation."""

    def test_create_databases(self, tmp_path: Path, storage_manager: StorageManager):
        """Test that databases are created with correct schema."""
        index_path = tmp_path / "index.db"
        work_path = tmp_path / "work.db"

        # Verify databases exist
        assert index_path.exists()
        assert work_path.exists()

        # Verify index tables exist
        with storage_manager.get_index_session() as index_session:
            assert index_session.query(Snapshot).count() == 0

        # Verify work tables exist
        with storage_manager.get_work_session() as work_session:
            assert work_session.query(Run).count() == 0

    def test_create_databases_with_memory(
        self, storage_manager: StorageManager, index_session, work_session
    ):
        """Test database creation with in-memory databases using StaticPool."""
        # Create StorageManager with memory databases
        # This would normally fail without proper engine setup
        # Verify tables exist
        assert index_session.query(Snapshot).count() == 0
        assert work_session.query(Run).count() == 0


class TestSchemaVersioning:
    """Test schema version checking and validation."""

    def test_new_index_db_sets_version(self, storage_manager: StorageManager):
        """Test that new index.db gets schema_version in Meta table."""
        # Check that version was set
        with storage_manager.get_index_session() as session:
            meta = session.query(IndexMeta).filter_by(key="schema_version").first()
            assert meta is not None
            assert meta.value == INDEX_SCHEMA_VERSION

    def test_new_work_db_sets_version(self, storage_manager: StorageManager):
        """Test that new work.db gets schema_version in Meta table."""
        # Check that version was set
        with storage_manager.get_work_session() as session:
            meta = session.query(WorkMeta).filter_by(key="schema_version").first()
            assert meta is not None
            assert meta.value == WORK_SCHEMA_VERSION

    def test_index_version_mismatch_raises_error(self):
        """Test that wrong index.db version raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.db"

            # Create database with wrong version
            engine = create_engine(f"sqlite:///{index_path}")
            IndexBase.metadata.create_all(engine)

            # Manually set wrong version
            from sqlalchemy.orm import sessionmaker

            Session = sessionmaker(bind=engine)
            session = Session()
            meta = IndexMeta(key="schema_version", value="0.0.0")
            session.add(meta)
            session.commit()
            session.close()

            # Now try to initialize StorageManager - should fail
            with pytest.raises(RuntimeError, match="index.db schema version mismatch"):
                StorageManager(Path(tmpdir))

    def test_work_version_mismatch_raises_error(self):
        """Test that wrong work.db version raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_path = Path(tmpdir) / "work.db"

            # Create database with wrong version
            engine = create_engine(f"sqlite:///{work_path}")
            WorkBase.metadata.create_all(engine)

            # Manually set wrong version
            from sqlalchemy.orm import sessionmaker

            Session = sessionmaker(bind=engine)
            session = Session()
            meta = WorkMeta(key="schema_version", value="0.0.0")
            session.add(meta)
            session.commit()
            session.close()

            # Now try to initialize StorageManager - should fail
            with pytest.raises(RuntimeError, match="work.db schema version mismatch"):
                StorageManager(Path(tmpdir))


class TestReferentialIntegrity:
    """Test referential integrity validation methods."""

    def test_validate_snapshot_exists(self, storage_manager: StorageManager, snapshot):
        """Test _validate_snapshot_exists method."""
        # Non-existent snapshot
        assert not storage_manager._validate_snapshot_exists(999)

        # Now it should exist
        assert storage_manager._validate_snapshot_exists(snapshot.id)

    def test_validate_node_exists(self, storage_manager: StorageManager, node, snapshot):
        """Test _validate_node_exists method."""
        # Should exist
        assert storage_manager._validate_node_exists(node.id, snapshot.id)

        # Should not exist with wrong snapshot
        assert not storage_manager._validate_node_exists(node.id, 999)

        # Should not exist with wrong node_id
        assert not storage_manager._validate_node_exists(999, snapshot.id)

    def test_check_snapshot_has_runs(self, storage_manager: StorageManager, snapshot, work_session):
        """Test _check_snapshot_has_runs method."""
        # No runs yet
        assert not storage_manager._check_snapshot_has_runs(snapshot.id)

        # Create a run
        RunFactory(
            id=snapshot.id,
        )

        # Now has runs
        assert storage_manager._check_snapshot_has_runs(snapshot.id)

    def test_validate_snapshot_id_matches_run(
        self, storage_manager: StorageManager, work_session: Session
    ):
        """Test _validate_snapshot_id_matches_run method."""
        # Create run with snapshot_id=1
        run = cast(
            Run,
            RunFactory(
                snapshot_id=1,
            ),
        )

        # Should match
        assert storage_manager._validate_snapshot_id_matches_run(1, run.id)

        # Should not match
        assert not storage_manager._validate_snapshot_id_matches_run(2, run.id)

        # Non-existent run
        assert not storage_manager._validate_snapshot_id_matches_run(1, 999)


class TestDeletion:
    """Test deletion methods with referential integrity."""

    def test_delete_snapshot_with_runs_fails(
        self, storage_manager: StorageManager, snapshot, run: Run
    ):
        """Test that snapshots with runs cannot be deleted."""
        # Should fail to delete
        with pytest.raises(ValueError, match="runs still reference it"):
            storage_manager.delete_snapshot(snapshot.id)

    def test_delete_run_allows_snapshot_deletion(
        self, storage_manager: StorageManager, snapshot, run: Run
    ):
        """Test that after deleting runs, snapshot can be deleted."""
        # Delete the run
        storage_manager.delete_run(run.id)

        # Now should be able to delete snapshot
        storage_manager.delete_snapshot(snapshot.id)

        # Verify it's gone
        with storage_manager.get_index_session() as session:
            assert session.query(Snapshot).filter_by(id=snapshot.id).first() is None


class TestImmutability:
    """Test snapshot immutability enforcement."""

    def test_node_modification_not_allowed(self, storage_manager: StorageManager):
        """Test that attempting to modify nodes raises NotImplementedError."""
        # Should raise error
        with pytest.raises(NotImplementedError, match="immutable"):
            storage_manager.update_node(1, name="new_name")

    def test_compute_features_externally_not_allowed(self, storage_manager: StorageManager):
        """Test that external call to compute_node_features raises error."""
        # Should raise error
        with pytest.raises(NotImplementedError, match="computed during ingest_filesystem"):
            storage_manager.compute_node_features(1)

    def test_read_only_session_prevents_mutation(self, storage_manager: StorageManager):
        """Test that read_only sessions prevent accidental mutations."""
        # Get read-only session
        with storage_manager.get_index_session(read_only=True) as session:
            # Try to add something
            new_snapshot = SnapshotFactory.build(
                created_at=datetime(2024, 1, 1, 0, 0, 0),
                root_path="/test",
                root_abs_path="/test",
            )
            session.add(new_snapshot)

            # Should fail on flush
            with pytest.raises(RuntimeError, match="read-only session"):
                session.flush()
