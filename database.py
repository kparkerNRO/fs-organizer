from pathlib import Path
from typing import Optional, List
from sqlalchemy import (
    TypeDecorator,
    create_engine,
    Column,
    Integer,
    String,
    Float,
    ForeignKey,
    event,
    inspect,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import json


class StringList(TypeDecorator):
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


class DictType(TypeDecorator):
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


Base = declarative_base()


class Folder(Base):
    __tablename__ = "folders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    folder_name = Column(String, nullable=False)
    folder_path = Column(String, nullable=False)
    parent_path = Column(String)
    depth = Column(Integer)
    cleaned_name = Column(String)
    categories = Column(StringList)  # Change this to use StringList type
    subject = Column(String)
    variants = Column(StringList)
    classification = Column(StringList)
    file_source = Column(String)
    num_folder_children = Column(Integer, default=0)
    num_file_children = Column(Integer, default=0)

    def __init__(self, **kwargs):
        # Convert lists to default empty lists if not provided
        kwargs["variants"] = kwargs.get("variants", [])
        kwargs["classification"] = kwargs.get("classification", [])
        kwargs["categories"] = kwargs.get(
            "categories", []
        )  # Add default for categories
        super(Folder, self).__init__(**kwargs)

    def __repr__(self):
        return f"Folder(id={self.id}, folder_name={self.folder_name}"


class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    depth = Column(Integer)

    def __repr__(self):
        return f"File(id={self.id}, file_name={self.file_name}"


class ProcessedName(Base):
    __tablename__ = "processed_names"

    id = Column(Integer, primary_key=True, autoincrement=True)
    group_id = Column(Integer, ForeignKey("groups.id"))
    folder_name = Column(String)
    grouped_name = Column(String)
    confidence = Column(Float)


class Group(Base):
    __tablename__ = "groups"

    id = Column(Integer, primary_key=True, autoincrement=True)
    group_name = Column(String)
    cannonical_name = Column(String)
    processed_names = relationship("ProcessedName", backref="group")


class FolderCategory(Base):
    __tablename__ = "folder_category"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    original_name = Column(String)
    classification = Column(String)

    folder_id = Column(Integer, ForeignKey("folders.id"), nullable=False)
    category_id = Column(Integer, ForeignKey("category.id"))


class Category(Base):
    __tablename__ = "category"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category_name = Column(String)
    classification = Column(String) # Description: the unified classifcation across all folders
    classification_counts = Column(DictType)
    total_count = Column(Integer)


def get_engine(db_path: Path):
    """Create and return a SQLAlchemy engine."""
    return create_engine(f"sqlite:///{db_path}")


def reset_tables(db_path: Path, tables: List[str]):
    """Reset tables in the database."""
    engine = get_engine(db_path)
    inspector = inspect(engine)

    # Drop existing tables if they exist
    table_names = [table.__tablename__ for table in tables]
    existing_tables = inspector.get_table_names()

    for table_name in table_names:
        if table_name in existing_tables:
            Base.metadata.tables[table_name].drop(engine)

    table_objs = [table.__table__ for table in tables]

    # Create new tables
    Base.metadata.create_all(engine, tables=table_objs)


def setup_category_summarization(db_path: Path):
    reset_tables(db_path, [Category])


def setup_gather(db_path: Path):
    """Create or open the SQLite database, ensuring tables exist."""
    reset_tables(db_path, [Folder, File])


def setup_group(db_path: Path):
    """Create or open the SQLite database for grouping functionality."""
    reset_tables(db_path, [ProcessedName, Group])


def setup_folder_categories(db_path: Path):
    """Create or open the SQLite database for categories."""
    reset_tables(db_path, [FolderCategory])


# Helper function to get a database session
def get_session(db_path: Path):
    """Create and return a new database session."""
    engine = get_engine(db_path)
    Session = sessionmaker(bind=engine)
    return Session()


# Example usage:
if __name__ == "__main__":
    db_path = Path("example.db")

    # Setup all tables
    setup_gather(db_path)
    setup_group(db_path)
    setup_folder_categories(db_path)

    # Example of using the session
    session = get_session(db_path)
    try:
        # Create a new folder
        new_folder = Folder(
            folder_name="Test Folder", folder_path="/test/path", depth=1
        )
        session.add(new_folder)
        session.commit()
    finally:
        session.close()
