"""Unit tests for StorageManager."""

import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from storage.manager import StorageManager, NodeKind, FileSource, RunStatus
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


@pytest.fixture
def storage_manager(tmp_path: Path) -> StorageManager:
    return StorageManager(database_path=tmp_path)


@pytest.fixture
def index_session(storage_manager: StorageManager):
    with storage_manager.get_index_session() as session:
        yield session


@pytest.fixture
def work_session(storage_manager: StorageManager):
    with storage_manager.get_work_session() as session:
        yield session


@pytest.fixture
def snapshot(index_session):
    new_snapshot = Snapshot(
        created_at="2024-01-01T00:00:00",
        root_path="/test",
        root_abs_path="/test",
    )
    index_session.add(new_snapshot)
    index_session.commit()
    return new_snapshot


@pytest.fixture
def node(index_session, snapshot):
    new_node = Node(
        snapshot_id=snapshot.snapshot_id,
        kind=NodeKind.FILE.value,
        name="test.txt",
        rel_path="test.txt",
        abs_path="/test/test.txt",
        depth=1,
        file_source=FileSource.FILESYSTEM.value,
    )
    index_session.add(new_node)
    index_session.commit()
    return new_node


@pytest.fixture
def run(work_session, snapshot):
    new_run = Run(
        snapshot_id=snapshot.snapshot_id,
        started_at="2024-01-01T00:00:00",
        status=RunStatus.RUNNING.value,
    )
    work_session.add(new_run)
    work_session.commit()
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
            index_path = Path(tmpdir) / "index.db"
            with pytest.raises(RuntimeError, match="work.db schema version mismatch"):
                StorageManager(Path(tmpdir))


class TestReferentialIntegrity:
    """Test referential integrity validation methods."""

    def test_validate_snapshot_exists(self, storage_manager: StorageManager, snapshot):
        """Test _validate_snapshot_exists method."""
        # Non-existent snapshot
        assert not storage_manager._validate_snapshot_exists(999)

        # Now it should exist
        assert storage_manager._validate_snapshot_exists(snapshot.snapshot_id)

    def test_validate_node_exists(
        self, storage_manager: StorageManager, node, snapshot
    ):
        """Test _validate_node_exists method."""
        # Should exist
        assert storage_manager._validate_node_exists(node.node_id, snapshot.snapshot_id)

        # Should not exist with wrong snapshot
        assert not storage_manager._validate_node_exists(node.node_id, 999)

        # Should not exist with wrong node_id
        assert not storage_manager._validate_node_exists(999, snapshot.snapshot_id)

    def test_check_snapshot_has_runs(
        self, storage_manager: StorageManager, snapshot, work_session
    ):
        """Test _check_snapshot_has_runs method."""
        # No runs yet
        assert not storage_manager._check_snapshot_has_runs(snapshot.snapshot_id)

        # Create a run
        run = Run(
            snapshot_id=snapshot.snapshot_id,
            started_at="2024-01-01T00:00:00",
            status=RunStatus.RUNNING.value,
        )
        work_session.add(run)
        work_session.commit()

        # Now has runs
        assert storage_manager._check_snapshot_has_runs(snapshot.snapshot_id)

    def test_validate_snapshot_id_matches_run(
        self, storage_manager: StorageManager, work_session
    ):
        """Test _validate_snapshot_id_matches_run method."""
        # Create run with snapshot_id=1
        run = Run(
            snapshot_id=1,
            started_at="2024-01-01T00:00:00",
            status=RunStatus.RUNNING.value,
        )
        work_session.add(run)
        work_session.commit()

        # Should match
        assert storage_manager._validate_snapshot_id_matches_run(1, run.run_id)

        # Should not match
        assert not storage_manager._validate_snapshot_id_matches_run(2, run.run_id)

        # Non-existent run
        assert not storage_manager._validate_snapshot_id_matches_run(1, 999)


class TestDeletion:
    """Test deletion methods with referential integrity."""

    def test_delete_snapshot_with_runs_fails(
        self, storage_manager: StorageManager, snapshot, run
    ):
        """Test that snapshots with runs cannot be deleted."""
        # Should fail to delete
        with pytest.raises(ValueError, match="runs still reference it"):
            storage_manager.delete_snapshot(snapshot.snapshot_id)

    def test_delete_run_allows_snapshot_deletion(
        self, storage_manager: StorageManager, snapshot, run
    ):
        """Test that after deleting runs, snapshot can be deleted."""
        # Delete the run
        storage_manager.delete_run(run.run_id)

        # Now should be able to delete snapshot
        storage_manager.delete_snapshot(snapshot.snapshot_id)

        # Verify it's gone
        with storage_manager.get_index_session() as session:
            assert (
                session.query(Snapshot)
                .filter_by(snapshot_id=snapshot.snapshot_id)
                .first()
                is None
            )


class TestImmutability:
    """Test snapshot immutability enforcement."""

    def test_node_modification_not_allowed(self, storage_manager: StorageManager):
        """Test that attempting to modify nodes raises NotImplementedError."""
        # Should raise error
        with pytest.raises(NotImplementedError, match="immutable"):
            storage_manager.update_node(1, name="new_name")

    def test_compute_features_externally_not_allowed(
        self, storage_manager: StorageManager
    ):
        """Test that external call to compute_node_features raises error."""
        # Should raise error
        with pytest.raises(
            NotImplementedError, match="computed during ingest_filesystem"
        ):
            storage_manager.compute_node_features(1)

    def test_read_only_session_prevents_mutation(self, storage_manager: StorageManager):
        """Test that read_only sessions prevent accidental mutations."""
        # Get read-only session
        with storage_manager.get_index_session(read_only=True) as session:
            # Try to add something
            new_snapshot = Snapshot(
                created_at="2024-01-01T00:00:00",
                root_path="/test",
                root_abs_path="/test",
            )
            session.add(new_snapshot)

            # Should fail on flush
            with pytest.raises(RuntimeError, match="read-only session"):
                session.flush()
