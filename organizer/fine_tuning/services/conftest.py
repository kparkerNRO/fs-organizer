"""Shared test fixtures for fine_tuning services tests"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from storage.index_models import IndexBase, Snapshot
from storage.training_models import LabelRun, TrainingBase


@pytest.fixture
def index_session():
    """Create in-memory test database for index data"""
    engine = create_engine("sqlite:///:memory:")
    IndexBase.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


@pytest.fixture
def training_session():
    """Create in-memory test database for training data"""
    engine = create_engine("sqlite:///:memory:")
    TrainingBase.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


@pytest.fixture
def label_run(training_session):
    """Create a test label run"""
    label_run = LabelRun(snapshot_id=1, label_source="test")
    training_session.add(label_run)
    training_session.flush()
    return label_run


@pytest.fixture
def sample_snapshot(index_session):
    """Create a test snapshot"""
    snapshot = Snapshot(
        snapshot_id=1,
        created_at="2024-01-01T00:00:00",
        root_path="/test",
        root_abs_path="/test",
    )
    index_session.add(snapshot)
    index_session.flush()
    return snapshot
