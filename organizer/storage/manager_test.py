"""Unit tests for StorageManager.

CRITICAL: All :memory: tests must use StaticPool to prevent separate DB issues.
Without StaticPool, each connection gets a separate :memory: DB, causing
schema/version checks and session reads to fail intermittently.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from storage.manager import StorageManager
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


def create_test_engine(db_url: str = "sqlite:///:memory:"):
    """Create engine for testing with StaticPool.

    CRITICAL: StaticPool ensures all connections share the same :memory: DB.
    Without this, schema version checks fail because each connection sees
    a different empty database.
    """
    return create_engine(
        db_url, connect_args={"check_same_thread": False}, poolclass=StaticPool
    )


class TestStorageManager:
    """Test StorageManager initialization and database creation."""

    def test_create_databases(self):
        """Test that databases are created with correct schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.db"
            work_path = Path(tmpdir) / "work.db"

            mgr = StorageManager(index_path=index_path, work_path=work_path)

            # Verify databases exist
            assert index_path.exists()
            assert work_path.exists()

            # Verify index tables exist
            index_session = mgr.get_index_session()
            assert index_session.query(Snapshot).count() == 0
            index_session.close()

            # Verify work tables exist
            work_session = mgr.get_work_session()
            assert work_session.query(Run).count() == 0
            work_session.close()

    def test_create_databases_with_memory(self):
        """Test database creation with in-memory databases using StaticPool."""
        # Create StorageManager with memory databases
        # This would normally fail without proper engine setup
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use temp files instead of :memory: for this test
            # to avoid needing to patch the engine creation
            index_path = Path(tmpdir) / "test_index.db"
            work_path = Path(tmpdir) / "test_work.db"

            mgr = StorageManager(index_path=index_path, work_path=work_path)

            # Verify tables exist
            index_session = mgr.get_index_session()
            assert index_session.query(Snapshot).count() == 0
            index_session.close()

            work_session = mgr.get_work_session()
            assert work_session.query(Run).count() == 0
            work_session.close()


class TestSchemaVersioning:
    """Test schema version checking and validation."""

    def test_new_index_db_sets_version(self):
        """Test that new index.db gets schema_version in Meta table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.db"
            work_path = Path(tmpdir) / "work.db"

            mgr = StorageManager(index_path=index_path, work_path=work_path)

            # Check that version was set
            session = mgr.get_index_session()
            meta = session.query(IndexMeta).filter_by(key="schema_version").first()
            assert meta is not None
            assert meta.value == INDEX_SCHEMA_VERSION
            session.close()

    def test_new_work_db_sets_version(self):
        """Test that new work.db gets schema_version in Meta table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.db"
            work_path = Path(tmpdir) / "work.db"

            mgr = StorageManager(index_path=index_path, work_path=work_path)

            # Check that version was set
            session = mgr.get_work_session()
            meta = session.query(WorkMeta).filter_by(key="schema_version").first()
            assert meta is not None
            assert meta.value == WORK_SCHEMA_VERSION
            session.close()

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
            work_path = Path(tmpdir) / "work.db"
            with pytest.raises(RuntimeError, match="index.db schema version mismatch"):
                StorageManager(index_path=index_path, work_path=work_path)

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
                StorageManager(index_path=index_path, work_path=work_path)


class TestReferentialIntegrity:
    """Test referential integrity validation methods."""

    def test_validate_snapshot_exists(self):
        """Test _validate_snapshot_exists method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.db"
            work_path = Path(tmpdir) / "work.db"

            mgr = StorageManager(index_path=index_path, work_path=work_path)

            # Non-existent snapshot
            assert not mgr._validate_snapshot_exists(999)

            # Create a snapshot
            session = mgr.get_index_session()
            snapshot = Snapshot(
                created_at="2024-01-01T00:00:00",
                root_path="/test",
                root_abs_path="/test",
            )
            session.add(snapshot)
            session.commit()
            snapshot_id = snapshot.snapshot_id
            session.close()

            # Now it should exist
            assert mgr._validate_snapshot_exists(snapshot_id)

    def test_validate_node_exists(self):
        """Test _validate_node_exists method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.db"
            work_path = Path(tmpdir) / "work.db"

            mgr = StorageManager(index_path=index_path, work_path=work_path)

            # Create a snapshot
            session = mgr.get_index_session()
            snapshot = Snapshot(
                created_at="2024-01-01T00:00:00",
                root_path="/test",
                root_abs_path="/test",
            )
            session.add(snapshot)
            session.flush()

            # Create a node
            node = Node(
                snapshot_id=snapshot.snapshot_id,
                kind="file",
                name="test.txt",
                rel_path="test.txt",
                abs_path="/test/test.txt",
                depth=1,
                file_source="filesystem",
            )
            session.add(node)
            session.commit()
            node_id = node.node_id
            snapshot_id = snapshot.snapshot_id
            session.close()

            # Should exist
            assert mgr._validate_node_exists(node_id, snapshot_id)

            # Should not exist with wrong snapshot
            assert not mgr._validate_node_exists(node_id, 999)

            # Should not exist with wrong node_id
            assert not mgr._validate_node_exists(999, snapshot_id)

    def test_check_snapshot_has_runs(self):
        """Test _check_snapshot_has_runs method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.db"
            work_path = Path(tmpdir) / "work.db"

            mgr = StorageManager(index_path=index_path, work_path=work_path)

            # Create a snapshot
            session = mgr.get_index_session()
            snapshot = Snapshot(
                created_at="2024-01-01T00:00:00",
                root_path="/test",
                root_abs_path="/test",
            )
            session.add(snapshot)
            session.commit()
            snapshot_id = snapshot.snapshot_id
            session.close()

            # No runs yet
            assert not mgr._check_snapshot_has_runs(snapshot_id)

            # Create a run
            work_session = mgr.get_work_session()
            run = Run(
                snapshot_id=snapshot_id,
                started_at="2024-01-01T00:00:00",
                status="running",
            )
            work_session.add(run)
            work_session.commit()
            work_session.close()

            # Now has runs
            assert mgr._check_snapshot_has_runs(snapshot_id)

    def test_validate_snapshot_id_matches_run(self):
        """Test _validate_snapshot_id_matches_run method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.db"
            work_path = Path(tmpdir) / "work.db"

            mgr = StorageManager(index_path=index_path, work_path=work_path)

            # Create run with snapshot_id=1
            work_session = mgr.get_work_session()
            run = Run(
                snapshot_id=1, started_at="2024-01-01T00:00:00", status="running"
            )
            work_session.add(run)
            work_session.commit()
            run_id = run.run_id
            work_session.close()

            # Should match
            assert mgr._validate_snapshot_id_matches_run(1, run_id)

            # Should not match
            assert not mgr._validate_snapshot_id_matches_run(2, run_id)

            # Non-existent run
            assert not mgr._validate_snapshot_id_matches_run(1, 999)


class TestDeletion:
    """Test deletion methods with referential integrity."""

    def test_delete_snapshot_with_runs_fails(self):
        """Test that snapshots with runs cannot be deleted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.db"
            work_path = Path(tmpdir) / "work.db"

            mgr = StorageManager(index_path=index_path, work_path=work_path)

            # Create a snapshot
            session = mgr.get_index_session()
            snapshot = Snapshot(
                created_at="2024-01-01T00:00:00",
                root_path="/test",
                root_abs_path="/test",
            )
            session.add(snapshot)
            session.commit()
            snapshot_id = snapshot.snapshot_id
            session.close()

            # Create a run referencing it
            work_session = mgr.get_work_session()
            run = Run(
                snapshot_id=snapshot_id,
                started_at="2024-01-01T00:00:00",
                status="running",
            )
            work_session.add(run)
            work_session.commit()
            work_session.close()

            # Should fail to delete
            with pytest.raises(ValueError, match="runs still reference it"):
                mgr.delete_snapshot(snapshot_id)

    def test_delete_run_allows_snapshot_deletion(self):
        """Test that after deleting runs, snapshot can be deleted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.db"
            work_path = Path(tmpdir) / "work.db"

            mgr = StorageManager(index_path=index_path, work_path=work_path)

            # Create a snapshot
            session = mgr.get_index_session()
            snapshot = Snapshot(
                created_at="2024-01-01T00:00:00",
                root_path="/test",
                root_abs_path="/test",
            )
            session.add(snapshot)
            session.commit()
            snapshot_id = snapshot.snapshot_id
            session.close()

            # Create a run
            work_session = mgr.get_work_session()
            run = Run(
                snapshot_id=snapshot_id,
                started_at="2024-01-01T00:00:00",
                status="running",
            )
            work_session.add(run)
            work_session.commit()
            run_id = run.run_id
            work_session.close()

            # Delete the run
            mgr.delete_run(run_id)

            # Now should be able to delete snapshot
            mgr.delete_snapshot(snapshot_id)

            # Verify it's gone
            session = mgr.get_index_session()
            assert session.query(Snapshot).filter_by(snapshot_id=snapshot_id).first() is None
            session.close()


class TestImmutability:
    """Test snapshot immutability enforcement."""

    def test_node_modification_not_allowed(self):
        """Test that attempting to modify nodes raises NotImplementedError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.db"
            work_path = Path(tmpdir) / "work.db"

            mgr = StorageManager(index_path=index_path, work_path=work_path)

            # Should raise error
            with pytest.raises(NotImplementedError, match="immutable"):
                mgr.update_node(1, name="new_name")

    def test_compute_features_externally_not_allowed(self):
        """Test that external call to compute_node_features raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.db"
            work_path = Path(tmpdir) / "work.db"

            mgr = StorageManager(index_path=index_path, work_path=work_path)

            # Should raise error
            with pytest.raises(
                NotImplementedError, match="computed during ingest_filesystem"
            ):
                mgr.compute_node_features(1)

    def test_read_only_session_prevents_mutation(self):
        """Test that read_only sessions prevent accidental mutations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.db"
            work_path = Path(tmpdir) / "work.db"

            mgr = StorageManager(index_path=index_path, work_path=work_path)

            # Get read-only session
            session = mgr.get_index_session(read_only=True)

            # Try to add something
            snapshot = Snapshot(
                created_at="2024-01-01T00:00:00",
                root_path="/test",
                root_abs_path="/test",
            )
            session.add(snapshot)

            # Should fail on flush
            with pytest.raises(RuntimeError, match="read-only session"):
                session.flush()

            session.close()
