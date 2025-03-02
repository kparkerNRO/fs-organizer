from pathlib import Path
from typing import List, Any
from sqlalchemy import (
    TypeDecorator,
    create_engine,
    Column,
    Integer,
    String,
    Float,
    ForeignKey,
    Boolean,
    DateTime,
    inspect,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import json
from datetime import datetime


Base = declarative_base()


class JsonList(TypeDecorator):
    """Custom type for handling lists stored as JSON strings"""

    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return []


class JsonDict(TypeDecorator):
    """Custom type for handling dictionaries stored as JSON strings"""

    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return {}


class Folder(Base):
    """Represents a filesystem folder with extracted metadata"""

    __tablename__ = "folders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    folder_name = Column(String, nullable=False, index=True)
    folder_path = Column(String, nullable=False, index=True)
    parent_path = Column(String, index=True)
    depth = Column(Integer)
    cleaned_name = Column(String)
    categories = Column(JsonList)
    subject = Column(String)
    variants = Column(JsonList)
    classification = Column(String)
    file_source = Column(String)
    num_folder_children = Column(Integer, default=0)
    num_file_children = Column(Integer, default=0)
    cleaned_path = Column(String)

    # # Relationships
    # partial_categories = relationship("PartialNameCategory", back_populates="folder")
    # group_entries = relationship("GroupCategoryEntry", back_populates="folder")

    def __init__(self, **kwargs):
        # Set default values for collections
        kwargs["variants"] = kwargs.get("variants", [])
        kwargs["classification"] = kwargs.get("classification", "UNKNOWN")
        kwargs["categories"] = kwargs.get("categories", [])
        super(Folder, self).__init__(**kwargs)

    def __repr__(self):
        return f"Folder(id={self.id}, name={self.folder_name})"


class File(Base):
    """Represents a file in the filesystem"""

    __tablename__ = "files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(String, nullable=False, index=True)
    file_path = Column(String, nullable=False, index=True)
    folder_id = Column(Integer, ForeignKey("folders.id"), nullable=True)
    depth = Column(Integer)

    # Relationship
    # folder = relationship("Folder")

    def __repr__(self):
        return f"File(id={self.id}, name={self.file_name})"


class PartialNameCategory(Base):
    """
    Represents a part of a folder name, once the string has been broken
    down into best-guess categories.
    (i.e. "garden indoor" -> "garden" and "indoor")
    """

    __tablename__ = "partial_name_categories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, index=True)
    original_name = Column(String)
    classification = Column(String)
    folder_id = Column(Integer, ForeignKey("folders.id"), nullable=False, index=True)
    hidden = Column(Boolean, default=False)
    confidence = Column(Float, default=1.0)

    # Relationships
    # folder = relationship("Folder", back_populates="partial_categories")
    # group_entries = relationship("GroupCategoryEntry", back_populates="partial_category")

    def __repr__(self):
        return f"PartialNameCategory(id={self.id}, name={self.name})"


class GroupingIteration(Base):
    """Tracks information about each grouping iteration"""

    __tablename__ = "grouping_iterations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    description = Column(String)
    parameters = Column(JsonDict)

    # Relationships
    # groups = relationship("GroupCategory", back_populates="iteration")

    def __repr__(self):
        return f"GroupingIteration(id={self.id}, timestamp={self.timestamp})"


class GroupCategory(Base):
    """Represents a group of related categories"""

    __tablename__ = "group_categories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, index=True)
    count = Column(Integer)
    group_confidence = Column(Float)
    iteration_id = Column(Integer, ForeignKey("grouping_iterations.id"), index=True)
    needs_review = Column(Boolean, default=False)
    reviewed = Column(Boolean, default=False)

    # Relationships
    # iteration = relationship("GroupingIteration", back_populates="groups")
    # entries = relationship("GroupCategoryEntry", back_populates="group")

    def __repr__(self):
        return f"GroupCategory(id={self.id}, name={self.name})"


class GroupCategoryEntry(Base):
    """Maps folders to groups through their partial name categories"""

    __tablename__ = "group_category_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    folder_id = Column(Integer, ForeignKey("folders.id"), nullable=False, index=True)
    partial_category_id = Column(
        Integer, ForeignKey("partial_name_categories.id"), index=True
    )
    group_id = Column(Integer, ForeignKey("group_categories.id"), index=True)
    iteration_id = Column(Integer, ForeignKey("grouping_iterations.id"), index=True)
    cluster_id = Column(Integer)
    processed_name = Column(String)
    pre_processed_name = Column(String)
    derived_names = Column(JsonList)
    path = Column(String, nullable=True)
    confidence = Column(Float, default=0)
    processed = Column(Boolean, default=False)

    # Relationships
    # folder = relationship("Folder", back_populates="group_entries")
    # partial_category = relationship("PartialNameCategory", back_populates="group_entries")
    # group = relationship("GroupCategory", back_populates="entries")

    def __repr__(self):
        return f"GroupCategoryEntry(id={self.id}, original={self.pre_processed_name})"


# class Category(Base):
#     """Represents a unified category across the system"""
#     __tablename__ = "categories"

#     id = Column(Integer, primary_key=True, autoincrement=True)
#     name = Column(String, unique=True, index=True)
#     classification = Column(String)
#     classification_counts = Column(JsonDict)
#     total_count = Column(Integer)

#     def __repr__(self):
#         return f"Category(id={self.id}, name={self.name})"


# Database management functions
def get_engine(db_path: Path):
    """Create and return a SQLAlchemy engine."""
    return create_engine(f"sqlite:///{db_path}")


def get_session(db_path: Path):
    """Create and return a new database session."""
    engine = get_engine(db_path)
    Session = sessionmaker(bind=engine)
    return Session()


def get_sessionmaker(db_path: Path):
    """Create and return a session factory."""
    engine = get_engine(db_path)
    return sessionmaker(bind=engine)


def reset_tables(
    db_path: Path, tables: List[Any] = None, legacy_tables: List[str] = None
):
    """Reset specified tables in the database."""
    if tables is None:
        tables = []
    if legacy_tables is None:
        legacy_tables = []

    engine = get_engine(db_path)
    inspector = inspect(engine)

    # Get table names
    table_names = [table.__tablename__ for table in tables] + legacy_tables
    existing_tables = inspector.get_table_names()

    # Drop existing tables
    for table_name in table_names:
        if table_name in existing_tables:
            if table_name in Base.metadata.tables:
                Base.metadata.tables[table_name].drop(engine)
            else:
                with engine.connect() as connection:
                    connection.execute(text(f"DROP TABLE {table_name}"))

    # Create new tables
    table_objs = [table.__table__ for table in tables]
    Base.metadata.create_all(engine, tables=table_objs)


# def setup_category_summarization(db_path: Path):
#     reset_tables(db_path, [Category])


def setup_gather(db_path: Path):
    """Create or open the SQLite database, ensuring tables exist."""
    reset_tables(db_path, [Folder, File])


def setup_group(db_path: Path):
    """Create or open the SQLite database for grouping functionality."""
    reset_tables(db_path, [GroupCategoryEntry])


def setup_folder_categories(db_path: Path):
    """Create or open the SQLite database for categories."""
    reset_tables(
        db_path,
        [PartialNameCategory, GroupCategory],
        legacy_tables=["folder_category", "group_record"],
    )


# Helper function to get a database session
# def get_session(db_path: Path):
#     """Create and return a new database session."""
#     engine = get_engine(db_path)
#     Session = sessionmaker(bind=engine)
#     return Session()

# def get_sessionmaker(db_path: Path):
#     """Create and return a new database session."""
#     engine = get_engine(db_path)
#     return sessionmaker(bind=engine)


# def get_latest_iteration(db_path: Path) -> Optional[GroupingIteration]:
#     """Get the most recent grouping iteration."""
#     with get_session(db_path) as session:
#         return session.query(GroupingIteration).order_by(GroupingIteration.id.desc()).first()
