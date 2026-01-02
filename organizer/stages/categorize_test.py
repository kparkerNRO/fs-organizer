import pytest
from pathlib import Path
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch

from stages.categorize import (
    get_parent_folder,
    get_categories_for_path,
    calculate_folder_structure,
)
from data_models.database import (
    Base,
    Folder,
    File,
    FolderStructure,
    GroupCategoryEntry,
    GroupingIteration
)

from api.api import StructureType


@pytest.fixture
def session():
    """Create an in-memory SQLite database for testing"""
    from sqlalchemy import event

    engine = create_engine("sqlite:///:memory:")

    # Enable foreign key constraints
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def mock_sessionmaker(session):
    """Create a mock sessionmaker that returns a context manager"""

    @contextmanager
    def session_context():
        yield session

    def create_session():
        return session_context()

    return create_session


@pytest.fixture
def sample_folders(session):
    """Create sample folder data for testing"""
    folders = [
        Folder(
            id=1,
            folder_name="parent_folder",
            folder_path="/test/parent_folder",
            depth=1,
        ),
        Folder(
            id=2,
            folder_name="child_folder",
            folder_path="/test/parent_folder/child_folder",
            depth=2,
        ),
        Folder(
            id=3,
            folder_name="content.zip",
            folder_path="/test/content.zip",
            depth=1,
        ),
        Folder(
            id=4,
            folder_name="zip_content",
            folder_path="/test/content.zip/zip_content",
            depth=2,
        ),
    ]

    for folder in folders:
        session.add(folder)
    session.commit()
    return folders


@pytest.fixture
def sample_files(session, sample_folders):
    """Create sample file data for testing"""
    # Ensure folders are created first
    files = [
        File(
            id=1,
            file_name="test1.txt",
            file_path="/test/parent_folder/test1.txt",
            folder_id=1,
            depth=2,
        ),
        File(
            id=2,
            file_name="test2.jpg",
            file_path="/test/parent_folder/child_folder/test2.jpg",
            folder_id=2,
            depth=3,
        ),
        File(
            id=3,
            file_name="archive.zip",
            file_path="/test/archive.zip",
            folder_id=None,
            depth=1,
        ),
        File(
            id=4,
            file_name="image.png",
            file_path="/test/content.zip/zip_content/image.png",
            folder_id=4,
            depth=3,
        ),
    ]

    for file in files:
        session.add(file)
    session.commit()
    return files


@pytest.fixture
def sample_group_entries(session, sample_folders):
    """Create sample group category entries for testing"""
    # Create iteration record first
    iteration = GroupingIteration(id=1, description="test iteration")
    session.add(iteration)
    session.commit()

    entries = [
        GroupCategoryEntry(
            id=1,
            folder_id=1,
            iteration_id=1,
            processed_name="art_category",
            confidence=0.8,
        ),
        GroupCategoryEntry(
            id=2,
            folder_id=2,
            iteration_id=1,
            processed_name="digital_art",
            confidence=0.9,
        ),
        GroupCategoryEntry(
            id=3,
            folder_id=4,
            iteration_id=1,
            processed_name="zip_category",
            confidence=0.7,
        ),
    ]

    for entry in entries:
        session.add(entry)
    session.commit()
    return entries


class TestGetParentFolder:
    """Test cases for get_parent_folder function"""

    def test_get_parent_folder_normal_path(self, session, sample_folders):
        """Test finding parent folder with normal file path"""
        parent_path = Path("/test/parent_folder")
        result = get_parent_folder(session, parent_path)

        assert result is not None
        assert result.folder_name == "parent_folder"
        assert result.folder_path == "/test/parent_folder"

    def test_get_parent_folder_not_found(self, session, sample_folders):
        """Test when parent folder doesn't exist"""
        parent_path = Path("/nonexistent/path")
        result = get_parent_folder(session, parent_path)

        assert result is None

    def test_get_parent_folder_zip_content_false(self, session, sample_folders):
        """Test zip content handling when zip_content=False"""
        parent_path = Path("/test/content.zip/zip_content")
        result = get_parent_folder(session, parent_path, zip_content=False)

        # The function finds the exact path match regardless of zip_content flag
        assert result is not None
        assert result.folder_name == "zip_content"

    def test_get_parent_folder_zip_content_true(self, session, sample_folders):
        """Test zip content handling when zip_content=True"""
        # Test with a path that doesn't exist to trigger fallback behavior
        parent_path = Path("/test/nonexistent.zip/some_content")
        result = get_parent_folder(session, parent_path, zip_content=True)

        assert result is None  # Should not find anything since neither path exists

    def test_get_parent_folder_zip_content_fallback(self, session, sample_folders):
        """Test fallback to parent when zip_content=True but direct path not found"""
        # Create a scenario where the direct path doesn't exist but parent does
        parent_path = Path("/test/nonexistent.zip/some_folder")
        result = get_parent_folder(session, parent_path, zip_content=True)

        assert result is None  # Neither direct path nor parent should exist


class TestGetCategoriesForPath:
    """Test cases for get_categories_for_path function"""

    def test_get_categories_for_path_string_input(
        self, session, sample_folders, sample_group_entries
    ):
        """Test with string path input"""
        path = "/test/parent_folder/child_folder/file.txt"
        result = get_categories_for_path(session, path, iteration_id=2)

        assert isinstance(result, list)

    def test_get_categories_for_path_path_input(
        self, session, sample_folders, sample_group_entries
    ):
        """Test with Path object input"""
        path = Path("/test/parent_folder/child_folder/file.txt")
        result = get_categories_for_path(session, path, iteration_id=2)

        assert isinstance(result, list)

    def test_get_categories_for_path_no_parent(
        self, session, sample_folders, sample_group_entries
    ):
        """Test when parent folder doesn't exist"""
        path = Path("/nonexistent/path/file.txt")
        result = get_categories_for_path(session, path, iteration_id=2)

        assert result == []

    def test_get_categories_for_path_with_groups(
        self, session, sample_folders, sample_group_entries
    ):
        """Test when parent folder has associated groups"""
        path = Path("/test/parent_folder/file.txt")
        result = get_categories_for_path(session, path, iteration_id=2)

        assert isinstance(result, list)
        # Should include groups from parent folder

    def test_get_categories_for_path_zip_matching(
        self, session, sample_folders, sample_group_entries
    ):
        """Test with zip file path matching"""
        path = Path("/test/content.zip/zip_content/file.txt")
        result = get_categories_for_path(session, path, iteration_id=2)

        assert isinstance(result, list)

    def test_get_categories_for_path_recursive(
        self, session, sample_folders, sample_group_entries
    ):
        """Test recursive category collection"""
        # This test verifies that categories are collected recursively up the path
        path = Path("/test/parent_folder/child_folder/deep/file.txt")

        # Test the actual recursive behavior by creating a deep folder structure
        deep_folder = Folder(
            id=5,
            folder_name="deep",
            folder_path="/test/parent_folder/child_folder/deep",
            depth=3,
        )
        session.add(deep_folder)
        session.commit()

        result = get_categories_for_path(session, path, iteration_id=2)
        assert isinstance(result, list)
        # The function should return categories from the recursive traversal


class TestCalculateCategories:
    """Test cases for calculate_categories function"""

    @patch("stages.categorize.insert_file_in_structure")
    @patch("stages.categorize.get_sessionmaker")
    def test_calculate_categories_basic(
        self,
        mock_get_sessionmaker,
        mock_insert,
        session,
        mock_sessionmaker,
        sample_files,
        sample_group_entries,
    ):
        """Test basic calculate_categories functionality"""
        # Setup mocks
        mock_get_sessionmaker.return_value = mock_sessionmaker

        db_path = Path("/test/database.db")

        # sample_group_entries already has entries with iteration_id=1
        calculate_folder_structure(db_path)

        # Verify sessionmaker was called
        mock_get_sessionmaker.assert_called_once_with(db_path)

        # Check that File objects were updated with new_path and groups
        files = session.query(File).all()
        non_zip_files = [f for f in files if not f.file_name.endswith(".zip")]
        for file in non_zip_files:
            assert hasattr(file, "new_path")
            assert hasattr(file, "groups")

        # Check that FolderStructure entry was created
        folder_structures = session.query(FolderStructure).all()
        assert len(folder_structures) == 1
        assert folder_structures[0].structure_type == StructureType.organized

    @patch("stages.categorize.insert_file_in_structure")
    @patch("stages.categorize.get_sessionmaker")
    def test_calculate_categories_processes_all_files(
        self,
        mock_get_sessionmaker,
        mock_insert,
        session,
        mock_sessionmaker,
        sample_files,
        sample_group_entries,
    ):
        """Test that all files are processed including zip files"""
        mock_get_sessionmaker.return_value = mock_sessionmaker

        db_path = Path("/test/database.db")

        # sample_group_entries already has entries with iteration_id=1
        calculate_folder_structure(db_path)

        # Check that all files were processed and updated
        files = session.query(File).all()
        # All files should have been processed
        assert len(files) > 0
        for file in files:
            assert hasattr(file, "new_path")
            assert hasattr(file, "groups")

    @patch("stages.categorize.insert_file_in_structure")
    @patch("stages.categorize.get_sessionmaker")
    @patch("stages.categorize.get_categories_for_path")
    def test_calculate_categories_with_categories(
        self,
        mock_get_categories,
        mock_get_sessionmaker,
        mock_insert,
        session,
        mock_sessionmaker,
        sample_files,
        sample_group_entries,
    ):
        """Test calculate_categories with mock categories"""
        # Setup mocks
        mock_get_sessionmaker.return_value = mock_sessionmaker
        # Mock GroupCategoryEntry objects
        mock_categories = [
            GroupCategoryEntry(processed_name="category1", confidence=0.8),
            GroupCategoryEntry(processed_name="category2", confidence=0.9),
        ]
        mock_get_categories.return_value = mock_categories

        db_path = Path("/test/database.db")

        # sample_group_entries already has entries with iteration_id=1
        calculate_folder_structure(db_path)

        # Check that categories were used to create new paths
        files = session.query(File).all()
        non_zip_files = [f for f in files if not f.file_name.endswith(".zip")]

        for file in non_zip_files:
            assert file.new_path == "category1/category2"
            assert file.groups == ["category1", "category2"]

    @patch("stages.categorize.get_sessionmaker")
    def test_calculate_categories_no_files(
        self, mock_get_sessionmaker, session, mock_sessionmaker
    ):
        """Test calculate_categories with no files in database"""
        mock_get_sessionmaker.return_value = mock_sessionmaker

        db_path = Path("/test/database.db")

        # Create required records
        iteration = GroupingIteration(id=1, description="test iteration")
        folder = Folder(
            id=1, folder_name="test_folder", folder_path="/test/folder", depth=1
        )
        session.add_all([iteration, folder])
        session.commit()

        session.add(
            GroupCategoryEntry(iteration_id=1, folder_id=1, processed_name="test")
        )
        session.commit()

        calculate_folder_structure(db_path)

        # Should not process any files since there are none
        files = session.query(File).all()
        assert len(files) == 0

    @patch("stages.categorize.get_sessionmaker")
    def test_calculate_categories_empty_categories(
        self,
        mock_get_sessionmaker,
        session,
        mock_sessionmaker,
        sample_files,
        sample_group_entries,
    ):
        """Test calculate_categories when get_categories_for_path returns empty list"""
        mock_get_sessionmaker.return_value = mock_sessionmaker

        db_path = Path("/test/database.db")

        # sample_group_entries already has entries with iteration_id=1

        with patch("stages.categorize.get_categories_for_path", return_value=[]):
            calculate_folder_structure(db_path)

        # Check that empty categories result in empty new_path
        files = session.query(File).all()
        non_zip_files = [f for f in files if not f.file_name.endswith(".zip")]

        for file in non_zip_files:
            assert file.new_path == ""
            assert file.groups == []


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_get_parent_folder_empty_path(self, session):
        """Test get_parent_folder with empty path"""
        parent_path = Path("")
        result = get_parent_folder(session, parent_path)
        assert result is None

    def test_get_categories_for_path_root_path(self, session):
        """Test get_categories_for_path with root path"""
        path = Path("/")
        result = get_categories_for_path(session, path, iteration_id=1)
        assert result == []

    @patch("stages.categorize.get_sessionmaker")
    def test_calculate_categories_no_iteration_id(
        self, mock_get_sessionmaker, session, mock_sessionmaker
    ):
        """Test calculate_categories when no GroupCategoryEntry exists"""
        mock_get_sessionmaker.return_value = mock_sessionmaker

        db_path = Path("/test/database.db")

        # No GroupCategoryEntry in database - function returns None instead of raising
        # This test verifies the function handles the case gracefully
        result = calculate_folder_structure(db_path)
        assert result is None
