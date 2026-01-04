"""Shared test fixtures for fine_tuning services tests"""

import os
import random
from datetime import datetime

import pytest
from faker import Faker
from storage.factories import (
    ClassificationFactory,
    FileMappingFactory,
    FileNodeFactory,
    FolderStructureFactory,
    GroupCategoryEntryFactory,
    GroupCategoryFactory,
    GroupEntryFactory,
    GroupIterationFactory,
    LabelRunFactory,
    ModelRunFactory,
    NodeFactory,
    PartialNameCategoryFactory,
    RunFactory,
    SamplePredictionFactory,
    SnapshotFactory,
    StageStateFactory,
    TrainingSampleFactory,
    WorkMetaFactory,
)
from storage.manager import StorageManager


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
    """Alias for StorageManager-backed work session."""
    return storage_work_session


@pytest.fixture
def index_session(storage_index_session):
    """Alias for StorageManager-backed index session."""
    return storage_index_session


@pytest.fixture
def storage_training_session(storage_manager):
    """Create a training session backed by StorageManager."""
    with storage_manager.get_training_session() as session:
        LabelRunFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        TrainingSampleFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        ModelRunFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        SamplePredictionFactory._meta.sqlalchemy_session = session  # type: ignore[misc]
        yield session


@pytest.fixture
def training_session(storage_training_session):
    """Alias for StorageManager-backed training session."""
    return storage_training_session


@pytest.fixture
def label_run(training_session):
    """Create a test label run using factory"""
    return LabelRunFactory()


@pytest.fixture
def sample_snapshot(index_session):
    """Create a test snapshot using factory"""
    return SnapshotFactory()


@pytest.fixture
def model_run(training_session):
    """Create a test model run using factory"""
    return ModelRunFactory()


@pytest.fixture
def storage_snapshot(storage_index_session):
    """Create a snapshot using StorageManager-backed session."""
    return SnapshotFactory()


@pytest.fixture
def storage_run(storage_work_session, storage_snapshot):
    """Create a Run tied to the storage snapshot."""
    return RunFactory(
        snapshot_id=storage_snapshot.id,
        started_at=datetime.now(),
    )


@pytest.fixture
def storage_iteration(storage_work_session, storage_run):
    """Create a GroupIteration for storage-backed tests."""
    return GroupIterationFactory(
        run=storage_run,
    )


@pytest.fixture
def sample_run(work_session):
    """Create a sample Run for testing."""
    return RunFactory(started_at=datetime.now())


@pytest.fixture
def sample_iteration(work_session, sample_run):
    """Create a sample GroupIteration for testing.

    Returns a GroupIteration tied to the sample run.
    """
    return GroupIterationFactory(
        run=sample_run,
    )
