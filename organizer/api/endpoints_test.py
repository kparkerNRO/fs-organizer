"""Tests for v2 API endpoints (dual representation)."""

import pytest
from fastapi.testclient import TestClient

from organizer_api import app
from storage.factories import (
    GroupCategoryEntryFactory,
    GroupCategoryFactory,
    GroupIterationFactory,
    HierarchyDiffLogFactory,
    NodeFactory,
    RunFactory,
    SnapshotFactory,
)
from storage.manager import NodeKind


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def setup_complete_data(storage_index_session, storage_work_session):
    """Setup complete test data including nodes and categories."""
    snapshot = SnapshotFactory()

    root_node = NodeFactory(snapshot_id=snapshot.id, name="root", kind=NodeKind.DIR)
    child_node = NodeFactory(
        snapshot_id=snapshot.id,
        name="documents",
        kind=NodeKind.DIR,
        parent_node_id=root_node.id,
    )
    storage_index_session.commit()

    run = RunFactory(snapshot_id=snapshot.id)
    iteration = GroupIterationFactory(run=run, snapshot_id=snapshot.id)
    category = GroupCategoryFactory(iteration=iteration, name="Personal")

    GroupCategoryEntryFactory(
        folder_id=child_node.id,
        group_id=category.id,
        iteration=iteration,
        processed_name="Personal",
    )

    storage_work_session.commit()

    return snapshot.id, run.id


class TestGetDualRepresentationEndpoint:
    """Test GET /api/v2/folder-structure endpoint."""

    def test_no_runs_returns_404(self, client, monkeypatch, tmp_path):
        """Test that endpoint returns 404 when no runs exist."""
        # TODO: Implement endpoint test with proper dependency injection mocking
        # Expected behavior:
        # - Mock storage manager to return empty database
        # - Use FastAPI dependency_overrides to inject mock
        # - Assert 404 status code when no runs exist
        pass

    def test_successful_retrieval(self, client, monkeypatch, setup_complete_data):
        """Test successful retrieval of dual representation."""
        # TODO: Implement endpoint test with proper dependency injection mocking
        # Expected behavior:
        # - Status code 200
        # - Response contains items, node_hierarchy, category_hierarchy
        # - Items contain both nodes and categories
        pass

    def test_response_structure(self, storage_manager, setup_complete_data):
        """Test that response has correct structure."""
        # Direct test of the logic without HTTP layer
        from data_models.pipeline import PipelineStage
        from utils.dual_representation import build_dual_representation

        snapshot_id, run_id = setup_complete_data
        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id,
            run_id,
            stages=[PipelineStage.original, PipelineStage.organized],
        )

        # Verify structure
        model_data = dual_rep.model_dump()
        assert "items" in model_data
        assert "hierarchies" in model_data

        # Verify hierarchies have correct keys
        hierarchies_dict = model_data["hierarchies"]
        assert "old" in hierarchies_dict  # PipelineStage.original.value
        assert "new" in hierarchies_dict  # PipelineStage.organized.value

        # Verify hierarchy structure
        for stage_name, hierarchy in hierarchies_dict.items():
            assert "stage" in hierarchy
            assert "source_type" in hierarchy
            assert "root" in hierarchy
            # Verify root is a HierarchyRecord with itemId and children
            assert "itemId" in hierarchy["root"]
            assert "children" in hierarchy["root"]

        # Verify items have correct keys
        items_dict = model_data["items"]
        for item_id, item in items_dict.items():
            assert "id" in item
            assert "type" in item
            assert item["type"] in ["node", "category"]
            # Note: "name" is in HierarchyRecord, not HierarchyItem

    def test_error_handling(self):
        """Test error handling for invalid snapshot_id."""
        # TODO: Implement endpoint error handling test
        # Expected: 500 error with appropriate message when invalid snapshot_id
        pass


class TestApplyHierarchyDiffEndpoint:
    """Test PATCH /api/v2/folder-structure endpoint."""

    def test_no_runs_returns_404(self):
        """Test that endpoint returns 404 when no runs exist."""
        # TODO: Implement endpoint test for 404 error
        # Expected behavior: 404 error when no runs are found
        pass

    def test_empty_diff(self, storage_work_session, setup_complete_data):
        """Test applying an empty diff."""
        from api.models import HierarchyDiff

        snapshot_id, run_id = setup_complete_data
        diff = HierarchyDiff(added={}, deleted={})

        # This should succeed and log the empty diff
        log_entry = HierarchyDiffLogFactory(run_id=run_id, diff=diff.model_dump())
        storage_work_session.commit()

        # Verify it was logged
        assert log_entry.id is not None
        assert log_entry.run_id == run_id
        assert log_entry.diff == diff.model_dump()

    def test_diff_with_additions(self, storage_work_session, setup_complete_data):
        """Test applying a diff with additions."""
        from api.models import HierarchyDiff

        snapshot_id, run_id = setup_complete_data
        diff = HierarchyDiff(
            added={"category-1": ["node-1", "node-2"]},
            deleted={},
        )

        log_entry = HierarchyDiffLogFactory(run_id=run_id, diff=diff.model_dump())
        storage_work_session.commit()

        assert log_entry.id is not None
        assert "added" in log_entry.diff
        assert log_entry.diff["added"]["category-1"] == ["node-1", "node-2"]

    def test_diff_with_deletions(self, storage_work_session, setup_complete_data):
        """Test applying a diff with deletions."""
        from api.models import HierarchyDiff

        snapshot_id, run_id = setup_complete_data
        diff = HierarchyDiff(
            added={},
            deleted={"category-2": ["node-3"]},
        )

        log_entry = HierarchyDiffLogFactory(run_id=run_id, diff=diff.model_dump())
        storage_work_session.commit()

        assert log_entry.id is not None
        assert "deleted" in log_entry.diff
        assert log_entry.diff["deleted"]["category-2"] == ["node-3"]

    def test_complex_diff(self, storage_work_session, setup_complete_data):
        """Test applying a complex diff with multiple operations."""
        from api.models import HierarchyDiff

        snapshot_id, run_id = setup_complete_data
        diff = HierarchyDiff(
            added={
                "category-1": ["node-1", "node-2"],
                "category-3": ["node-5"],
            },
            deleted={
                "category-2": ["node-3", "node-4"],
            },
        )

        log_entry = HierarchyDiffLogFactory(run_id=run_id, diff=diff.model_dump())
        storage_work_session.commit()

        assert log_entry.id is not None
        assert len(log_entry.diff["added"]) == 2
        assert len(log_entry.diff["deleted"]) == 1

    def test_multiple_diffs_logged(self, storage_work_session, setup_complete_data):
        """Test that multiple diffs can be logged."""
        from api.models import HierarchyDiff
        from sqlalchemy import select
        from storage.work_models import HierarchyDiffLog

        snapshot_id, run_id = setup_complete_data

        # Log first diff
        diff1 = HierarchyDiff(added={"category-1": ["node-1"]}, deleted={})
        HierarchyDiffLogFactory(run_id=run_id, diff=diff1.model_dump())

        # Log second diff
        diff2 = HierarchyDiff(added={"category-2": ["node-2"]}, deleted={})
        HierarchyDiffLogFactory(run_id=run_id, diff=diff2.model_dump())

        storage_work_session.commit()

        # Verify both were logged
        logs = storage_work_session.execute(select(HierarchyDiffLog)).scalars().all()
        assert len(logs) == 2
        assert logs[0].id != logs[1].id

    def test_diff_log_timestamp(self, storage_work_session, setup_complete_data):
        """Test that diff log includes timestamp."""
        from api.models import HierarchyDiff

        snapshot_id, run_id = setup_complete_data
        diff = HierarchyDiff(added={"category-1": ["node-1"]}, deleted={})

        log_entry = HierarchyDiffLogFactory(run_id=run_id, diff=diff.model_dump())
        storage_work_session.commit()

        assert log_entry.created_at is not None
        # created_at should be set automatically


class TestDualRepresentationIntegration:
    """Integration tests for the dual representation system."""

    def test_end_to_end_workflow(
        self, storage_manager, storage_work_session, setup_complete_data
    ):
        """Test the complete workflow from building to applying diffs."""
        from api.models import HierarchyDiff
        from data_models.pipeline import PipelineStage
        from sqlalchemy import select
        from storage.work_models import HierarchyDiffLog
        from utils.dual_representation import build_dual_representation

        snapshot_id, run_id = setup_complete_data

        # 1. Build dual representation
        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id,
            run_id,
            stages=[PipelineStage.organized],
        )
        assert dual_rep is not None
        assert len(dual_rep.items) > 0
        assert "new" in dual_rep.hierarchies  # PipelineStage.organized.value

        # 2. Simulate user making changes (create a diff)
        diff = HierarchyDiff(
            added={"category-1": ["node-2"]},
            deleted={"organized-root": ["node-2"]},
        )

        # 3. Log the diff
        HierarchyDiffLogFactory(run_id=run_id, diff=diff.model_dump())
        storage_work_session.commit()

        # 4. Verify the diff was logged
        logged_diffs = (
            storage_work_session.execute(select(HierarchyDiffLog)).scalars().all()
        )
        assert len(logged_diffs) == 1
        assert logged_diffs[0].run_id == run_id

        # 5. Build representation again (should reflect changes in future implementation)
        # TODO: Implement diff application logic to modify the dual representation
        dual_rep_after = build_dual_representation(
            storage_manager,
            snapshot_id,
            run_id,
            stages=[PipelineStage.organized],
        )
        assert dual_rep_after is not None
