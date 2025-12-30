"""Shared test fixtures for fine_tuning services tests"""

import os
import random

import pytest
from faker import Faker
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from storage.index_models import IndexBase
from storage.training_models import TrainingBase
from fine_tuning.services.factories import (
    LabelRunFactory,
    ModelRunFactory,
    NodeFactory,
    SamplePredictionFactory,
    SnapshotFactory,
    TrainingSampleFactory,
)


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


@pytest.fixture
def model_run(training_session):
    """Create a test model run using factory"""
    return ModelRunFactory(run_id=1)
