"""Shared test fixtures for fine_tuning services tests"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from storage.index_models import IndexBase
from storage.training_models import TrainingBase

from .factories import (
    LabelRunFactory,
    ModelRunFactory,
    NodeFactory,
    SamplePredictionFactory,
    SnapshotFactory,
    TrainingSampleFactory,
)


@pytest.fixture
def index_session():
    """Create in-memory test database for index data"""
    engine = create_engine("sqlite:///:memory:")
    IndexBase.metadata.create_all(engine)

    with Session(engine) as session:
        # Configure factories to use this session
        NodeFactory._meta.sqlalchemy_session = session
        SnapshotFactory._meta.sqlalchemy_session = session
        yield session


@pytest.fixture
def training_session():
    """Create in-memory test database for training data"""
    engine = create_engine("sqlite:///:memory:")
    TrainingBase.metadata.create_all(engine)

    with Session(engine) as session:
        # Configure factories to use this session
        LabelRunFactory._meta.sqlalchemy_session = session
        TrainingSampleFactory._meta.sqlalchemy_session = session
        ModelRunFactory._meta.sqlalchemy_session = session
        SamplePredictionFactory._meta.sqlalchemy_session = session
        yield session


@pytest.fixture
def label_run(training_session):
    """Create a test label run using factory"""
    return LabelRunFactory(snapshot_id=1)


@pytest.fixture
def sample_snapshot(index_session):
    """Create a test snapshot using factory"""
    return SnapshotFactory(snapshot_id=1)
