"""Training database manager.

Provides utilities for creating and managing the training.db database,
including initialization, schema management, and session creation.
"""

from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from storage.training_models import TrainingBase, Meta, TRAINING_SCHEMA_VERSION


def init_training_db(db_path: Path) -> None:
    """
    Initialize a new training database with schema.

    Args:
        db_path: Path where the training.db file should be created
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{db_path}")

    # Create all tables
    TrainingBase.metadata.create_all(engine)

    # Store schema version in meta table
    with Session(engine) as session:
        version_meta = Meta(key="schema_version", value=TRAINING_SCHEMA_VERSION)
        session.merge(version_meta)
        session.commit()

    engine.dispose()


def get_training_session(db_path: Path) -> Session:
    """
    Get a SQLAlchemy session for the training database.

    Args:
        db_path: Path to the training.db file

    Returns:
        SQLAlchemy Session instance
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"Training database not found at {db_path}. "
            "Call init_training_db() first to create it."
        )

    engine = create_engine(f"sqlite:///{db_path}")
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def get_schema_version(db_path: Path) -> str | None:
    """
    Get the schema version from the training database.

    Args:
        db_path: Path to the training.db file

    Returns:
        Schema version string or None if not set
    """
    if not db_path.exists():
        return None

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        result = session.execute(
            text("SELECT value FROM meta WHERE key = 'schema_version'")
        ).fetchone()
        engine.dispose()
        return result[0] if result else None


def get_or_create_training_session(db_path: Path) -> Session:
    """
    Get a session for the training database, creating it if it doesn't exist.

    Args:
        db_path: Path to the training.db file

    Returns:
        SQLAlchemy Session instance
    """
    if not db_path.exists():
        init_training_db(db_path)

    return get_training_session(db_path)
