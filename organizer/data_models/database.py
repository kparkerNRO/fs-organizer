from pathlib import Path
from typing import List, Any, Optional
from sqlalchemy import (
    TypeDecorator,
    create_engine,
    String,
    Float,
    ForeignKey,
    DateTime,
    inspect,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    sessionmaker,
    Mapped,
    mapped_column,
)
import json
from datetime import datetime


class Base(DeclarativeBase):
    pass


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

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    folder_name: Mapped[str] = mapped_column(String, index=True)
    folder_path: Mapped[str] = mapped_column(String, index=True)
    parent_path: Mapped[Optional[str]] = mapped_column(String, index=True)
    depth: Mapped[Optional[int]]
    cleaned_name: Mapped[Optional[str]]
    categories: Mapped[Optional[List]] = mapped_column(JsonList, default=[])
    subject: Mapped[Optional[str]]
    variants: Mapped[Optional[List]] = mapped_column(JsonList, default=[])
    classification: Mapped[Optional[str]] = mapped_column(default="UNKNOWN")
    file_source: Mapped[Optional[str]]
    num_folder_children: Mapped[int] = mapped_column(default=0)
    num_file_children: Mapped[int] = mapped_column(default=0)
    cleaned_path: Mapped[Optional[str]]

    # # Relationships
    # partial_categories: Mapped[List["PartialNameCategory"]] = relationship(back_populates="folder")
    # group_entries: Mapped[List["GroupCategoryEntry"]] = relationship(back_populates="folder")

    def __repr__(self):
        return f"Folder(id={self.id}, name={self.folder_name})"


class File(Base):
    """Represents a file in the filesystem"""

    __tablename__ = "files"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    file_name: Mapped[str] = mapped_column(String, index=True)
    file_path: Mapped[str] = mapped_column(String, index=True)
    folder_id: Mapped[Optional[int]] = mapped_column(ForeignKey("folders.id"))
    depth: Mapped[Optional[int]]

    parent_folder_id: Mapped[Optional[int]] = mapped_column(ForeignKey("folders.id"))
    groups: Mapped[Optional[List]] = mapped_column(JsonList)
    original_path: Mapped[Optional[str]]
    new_path: Mapped[Optional[str]]

    # Relationship
    # folder: Mapped[Optional["Folder"]] = relationship()

    def __repr__(self):
        return f"File(id={self.id}, name={self.file_name})"


class FolderStructure(Base):
    """
    Represents the most-recently calculated folder structure
    for the old and new structures. This should be serializable into
    data_models.api.FolderV2 and data_models.api.File objects via
    pydantic
    """

    __tablename__ = "folder_structure"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    structure_type: Mapped[str] = mapped_column(String)
    structure: Mapped[dict] = mapped_column(JsonDict)


class PartialNameCategory(Base):
    """
    Represents a part of a folder name, once the string has been broken
    down into best-guess categories.
    (i.e. "garden indoor" -> "garden" and "indoor")
    """

    __tablename__ = "partial_name_categories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[Optional[str]] = mapped_column(String, index=True)
    original_name: Mapped[Optional[str]]
    classification: Mapped[Optional[str]]
    folder_id: Mapped[int] = mapped_column(ForeignKey("folders.id"), index=True)
    hidden: Mapped[bool] = mapped_column(default=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)

    # Relationships
    # folder: Mapped["Folder"] = relationship(back_populates="partial_categories")
    # group_entries: Mapped[List["GroupCategoryEntry"]] = relationship(back_populates="partial_category")

    def __repr__(self):
        return f"PartialNameCategory(id={self.id}, name={self.name})"


class GroupingIteration(Base):
    """Tracks information about each grouping iteration"""

    __tablename__ = "grouping_iterations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    description: Mapped[Optional[str]]
    parameters: Mapped[Optional[dict]] = mapped_column(JsonDict)

    # Relationships
    # groups: Mapped[List["GroupCategory"]] = relationship(back_populates="iteration")

    def __repr__(self):
        return f"GroupingIteration(id={self.id}, timestamp={self.timestamp})"


class GroupCategory(Base):
    """Represents a group of related categories"""

    __tablename__ = "group_categories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[Optional[str]] = mapped_column(String, index=True)
    count: Mapped[Optional[int]]
    group_confidence: Mapped[Optional[float]] = mapped_column(Float)
    iteration_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("grouping_iterations.id"), index=True
    )
    needs_review: Mapped[bool] = mapped_column(default=False)
    reviewed: Mapped[bool] = mapped_column(default=False)

    # Relationships
    # iteration: Mapped[Optional["GroupingIteration"]] = relationship(back_populates="groups")
    # entries: Mapped[List["GroupCategoryEntry"]] = relationship(back_populates="group")

    def __repr__(self):
        return f"GroupCategory(id={self.id}, name={self.name})"


class GroupCategoryEntry(Base):
    """Maps folders to groups through their partial name categories"""

    __tablename__ = "group_category_entries"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    folder_id: Mapped[int] = mapped_column(ForeignKey("folders.id"), index=True)
    partial_category_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("partial_name_categories.id"), index=True
    )
    group_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("group_categories.id"), index=True
    )
    iteration_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("grouping_iterations.id"), index=True
    )
    cluster_id: Mapped[Optional[int]]
    processed_name: Mapped[Optional[str]]
    pre_processed_name: Mapped[Optional[str]]
    derived_names: Mapped[Optional[List]] = mapped_column(JsonList)
    path: Mapped[Optional[str]]
    confidence: Mapped[float] = mapped_column(Float, default=0)
    processed: Mapped[bool] = mapped_column(default=False)

    # Relationships
    # folder: Mapped["Folder"] = relationship(back_populates="group_entries")
    # partial_category: Mapped[Optional["PartialNameCategory"]] = relationship(back_populates="group_entries")
    # group: Mapped[Optional["GroupCategory"]] = relationship(back_populates="entries")

    def __repr__(self):
        return f"GroupCategoryEntry(id={self.id}, original={self.pre_processed_name}, processed={self.processed_name})"


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


def setup_gather(db_path: Path):
    """Create or open the SQLite database, ensuring tables exist."""
    reset_tables(db_path, [Folder, File, FolderStructure])


def setup_group(db_path: Path):
    """Create or open the SQLite database for grouping functionality."""
    reset_tables(db_path, [GroupingIteration, GroupCategoryEntry])


def setup_folder_categories(db_path: Path):
    """Create or open the SQLite database for categories."""
    reset_tables(
        db_path,
        [PartialNameCategory, GroupCategory],
        legacy_tables=["folder_category", "group_record"],
    )
