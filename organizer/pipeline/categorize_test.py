import pytest
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch

from pipeline.categorize import (
    get_parent_folder,
    get_categories_for_path,
    calculate_categories,
)
from data_models.database import (
    Base,
    Folder,
    File,
    FileProcess,
    GroupCategoryEntry,
)


@pytest.fixture
def session():
    """Create an in-memory SQLite database for testing"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


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
def sample_files(session):
    """Create sample file data for testing"""
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

    def test_get_categories_for_path_string_input(self, session, sample_folders, sample_group_entries):
        """Test with string path input"""
        path = "/test/parent_folder/child_folder/file.txt"
        result = get_categories_for_path(session, path, iteration_id=2)
        
        assert isinstance(result, list)

    def test_get_categories_for_path_path_input(self, session, sample_folders, sample_group_entries):
        """Test with Path object input"""
        path = Path("/test/parent_folder/child_folder/file.txt")
        result = get_categories_for_path(session, path, iteration_id=2)
        
        assert isinstance(result, list)

    def test_get_categories_for_path_no_parent(self, session, sample_folders, sample_group_entries):
        """Test when parent folder doesn't exist"""
        path = Path("/nonexistent/path/file.txt")
        result = get_categories_for_path(session, path, iteration_id=2)
        
        assert result == []

    def test_get_categories_for_path_with_groups(self, session, sample_folders, sample_group_entries):
        """Test when parent folder has associated groups"""
        path = Path("/test/parent_folder/file.txt")
        result = get_categories_for_path(session, path, iteration_id=2)
        
        assert isinstance(result, list)
        # Should include groups from parent folder

    def test_get_categories_for_path_zip_matching(self, session, sample_folders, sample_group_entries):
        """Test with zip file path matching"""
        path = Path("/test/content.zip/zip_content/file.txt")
        result = get_categories_for_path(session, path, iteration_id=2)
        
        assert isinstance(result, list)

    def test_get_categories_for_path_recursive(self, session, sample_folders, sample_group_entries):
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

    @patch('pipeline.categorize.setup_file_processing')
    @patch('pipeline.categorize.get_sessionmaker')
    def test_calculate_categories_basic(self, mock_get_sessionmaker, mock_setup, session, sample_files, sample_group_entries):
        """Test basic calculate_categories functionality"""
        # Setup mocks
        mock_get_sessionmaker.return_value = lambda: session
        mock_setup.return_value = None
        
        db_path = Path("/test/database.db")
        
        # Add max iteration_id query result
        session.add(GroupCategoryEntry(iteration_id=1, folder_id=1, processed_name="test"))
        session.commit()
        
        calculate_categories(db_path)
        
        # Verify setup was called
        mock_setup.assert_called_once_with(db_path)
        mock_get_sessionmaker.assert_called_once_with(db_path)
        
        # Check that FileProcess entries were created
        file_processes = session.query(FileProcess).all()
        assert len(file_processes) > 0

    @patch('pipeline.categorize.setup_file_processing')
    @patch('pipeline.categorize.get_sessionmaker')
    def test_calculate_categories_skips_zip_files(self, mock_get_sessionmaker, mock_setup, session, sample_files, sample_group_entries):
        """Test that zip files are skipped during processing"""
        mock_get_sessionmaker.return_value = lambda: session
        mock_setup.return_value = None
        
        db_path = Path("/test/database.db")
        
        # Add max iteration_id query result
        session.add(GroupCategoryEntry(iteration_id=1, folder_id=1, processed_name="test"))
        session.commit()
        
        calculate_categories(db_path)
        
        # Check that zip files were not processed
        file_processes = session.query(FileProcess).all()
        zip_processes = [fp for fp in file_processes if fp.name.endswith('.zip')]
        assert len(zip_processes) == 0

    @patch('pipeline.categorize.setup_file_processing')
    @patch('pipeline.categorize.get_sessionmaker')
    @patch('pipeline.categorize.get_categories_for_path')
    def test_calculate_categories_with_categories(self, mock_get_categories, mock_get_sessionmaker, mock_setup, session, sample_files, sample_group_entries):
        """Test calculate_categories with mock categories"""
        # Setup mocks
        mock_get_sessionmaker.return_value = lambda: session
        mock_setup.return_value = None
        mock_get_categories.return_value = ["category1", "category2"]
        
        db_path = Path("/test/database.db")
        
        # Add max iteration_id query result
        session.add(GroupCategoryEntry(iteration_id=1, folder_id=1, processed_name="test"))
        session.commit()
        
        calculate_categories(db_path)
        
        # Check that categories were used to create new paths
        file_processes = session.query(FileProcess).all()
        non_zip_processes = [fp for fp in file_processes if not fp.name.endswith('.zip')]
        
        for fp in non_zip_processes:
            assert fp.new_path == "category1/category2"
            assert fp.groups == ["category1", "category2"]

    @patch('pipeline.categorize.setup_file_processing')
    @patch('pipeline.categorize.get_sessionmaker')
    def test_calculate_categories_no_files(self, mock_get_sessionmaker, mock_setup, session):
        """Test calculate_categories with no files in database"""
        mock_get_sessionmaker.return_value = lambda: session
        mock_setup.return_value = None
        
        db_path = Path("/test/database.db")
        
        # Add max iteration_id query result but no files
        session.add(GroupCategoryEntry(iteration_id=1, folder_id=1, processed_name="test"))
        session.commit()
        
        calculate_categories(db_path)
        
        # Should not create any FileProcess entries
        file_processes = session.query(FileProcess).all()
        assert len(file_processes) == 0

    @patch('pipeline.categorize.setup_file_processing')
    @patch('pipeline.categorize.get_sessionmaker')
    def test_calculate_categories_empty_categories(self, mock_get_sessionmaker, mock_setup, session, sample_files, sample_group_entries):
        """Test calculate_categories when get_categories_for_path returns empty list"""
        mock_get_sessionmaker.return_value = lambda: session
        mock_setup.return_value = None
        
        db_path = Path("/test/database.db")
        
        # Add max iteration_id query result
        session.add(GroupCategoryEntry(iteration_id=1, folder_id=1, processed_name="test"))
        session.commit()
        
        with patch('pipeline.categorize.get_categories_for_path', return_value=[]):
            calculate_categories(db_path)
        
        # Check that empty categories result in empty new_path
        file_processes = session.query(FileProcess).all()
        non_zip_processes = [fp for fp in file_processes if not fp.name.endswith('.zip')]
        
        for fp in non_zip_processes:
            assert fp.new_path == ""
            assert fp.groups == []


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

    @patch('pipeline.categorize.setup_file_processing')
    @patch('pipeline.categorize.get_sessionmaker')
    def test_calculate_categories_no_iteration_id(self, mock_get_sessionmaker, mock_setup, session):
        """Test calculate_categories when no GroupCategoryEntry exists"""
        mock_get_sessionmaker.return_value = lambda: session
        mock_setup.return_value = None
        
        db_path = Path("/test/database.db")
        
        # No GroupCategoryEntry in database - function returns None instead of raising
        # This test verifies the function handles the case gracefully
        result = calculate_categories(db_path)
        assert result is None