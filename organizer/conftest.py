"""Shared test fixtures for fine_tuning services tests"""

from datetime import datetime
import os
import random

import pytest
from faker import Faker
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from storage.factories import (
    ClassificationFactory,
    FileMappingFactory,
    FolderStructureFactory,
    GroupCategoryEntryFactory,
    GroupCategoryFactory,
    GroupEntryFactory,
    GroupIterationFactory,
    LabelRunFactory,
    ModelRunFactory,
    NodeFactory,
    FileNodeFactory,
    RunFactory,
    SamplePredictionFactory,
    SnapshotFactory,
    StageStateFactory,
    TrainingSampleFactory,
    WorkMetaFactory,
    PartialNameCategoryFactory,
)
from storage.index_models import IndexBase
from storage.manager import StorageManager
from storage.training_models import TrainingBase
from storage.work_models import WorkBase


@pytest.fixture(scope="session", autouse=True)
def setup_factory_seed():
    """Configure factory_boy/Faker to use a deterministic seed for reproducibility.

    The seed can be set via FACTORY_SEED environment variable, or will be
    randomly generated. The seed is printed to stdout for reproducibility.
    """
    seed = os.environ.get("FACTORY_SEED")
    if seed:
        seed = int(seed)
    else:
        seed = random.randint(0, 2**32 - 1)

    print(f"\n{'=' * 70}")
    print(f"Factory seed: {seed}")
    print(f"To reproduce this test run, set: FACTORY_SEED={seed}")
    print(f"{'=' * 70}\n")

    Faker.seed(seed)
    random.seed(seed)

    return seed


@pytest.fixture
def storage_manager(tmp_path):
    """Create a StorageManager with temporary databases."""
    return StorageManager(database_path=tmp_path, initialize_training=True)


@pytest.fixture
def storage_index_session(storage_manager):
    """Create an index session backed by StorageManager."""
    with storage_manager.get_index_session() as session:
        NodeFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        FileNodeFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        SnapshotFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        yield session


@pytest.fixture
def storage_work_session(storage_manager):
    """Create a work session backed by StorageManager."""
    with storage_manager.get_work_session() as session:
        RunFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        StageStateFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        GroupIterationFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        GroupEntryFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        GroupCategoryFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        GroupCategoryEntryFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        PartialNameCategoryFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        ClassificationFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        FolderStructureFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        FileMappingFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        WorkMetaFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        yield session


@pytest.fixture
def work_session(storage_work_session):
    """Create an in-memory test database for work data (runs, iterations, etc.)

    Note: Some tests may have their own session fixture that includes both
    Base and WorkBase tables. In those cases, use that session instead.
    """
    engine = create_engine("sqlite:///:memory:")
    WorkBase.metadata.create_all(engine)

    with Session(engine) as session:
        RunFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        StageStateFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        GroupIterationFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        GroupEntryFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        GroupCategoryFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        GroupCategoryEntryFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        PartialNameCategoryFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        ClassificationFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        FolderStructureFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        FileMappingFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        WorkMetaFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        yield session


@pytest.fixture
def index_session():
    """Create in-memory test database for index data."""
    engine = create_engine("sqlite:///:memory:")
    IndexBase.metadata.create_all(engine)

    with Session(engine) as session:
        NodeFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        FileNodeFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        SnapshotFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        yield session


@pytest.fixture
def training_session():
    """Create in-memory test database for training data"""

    engine = create_engine("sqlite:///:memory:")
    TrainingBase.metadata.create_all(engine)

    with Session(engine) as session:
        # Configure factories to use this session
        LabelRunFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        TrainingSampleFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        ModelRunFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        SamplePredictionFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        yield session


@pytest.fixture
def label_run(training_session):
    """Create a test label run using factory"""
    return LabelRunFactory(snapshot_id=1)


@pytest.fixture
def sample_snapshot(index_session):
    """Create a test snapshot using factory"""
    return SnapshotFactory(snapshot_id=1)


@pytest.fixture
def model_run(training_session):
    """Create a test model run using factory"""
    return ModelRunFactory(run_id=1)


@pytest.fixture
def storage_snapshot(storage_index_session):
    """Create a snapshot using StorageManager-backed session."""
    return SnapshotFactory(snapshot_id=1)


@pytest.fixture
def storage_run(storage_work_session, storage_snapshot):
    """Create a Run tied to the storage snapshot."""
    return RunFactory(
        snapshot_id=storage_snapshot.snapshot_id,
        started_at=datetime.now().isoformat(),
    )


@pytest.fixture
def storage_iteration(storage_work_session, storage_run):
    """Create a GroupIteration for storage-backed tests."""
    return GroupIterationFactory(
        description="test iteration",
        run=storage_run,
    )


@pytest.fixture
def sample_run(work_session):
    """Create a sample Run for testing."""
    return RunFactory(snapshot_id=1, started_at=datetime.now().isoformat())


@pytest.fixture
def sample_iteration(work_session, sample_run):
    """Create a sample GroupIteration for testing.

    Returns a GroupIteration with id=1, run_id=1, snapshot_id=1.
    """
    return GroupIterationFactory(
        run=sample_run,
        description="test iteration",
    )
