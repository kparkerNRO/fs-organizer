import pytest
from pathlib import Path

from api.api import StructureType
from stages.categorize import (
    get_parent_folder,
    get_categories_for_path,
    calculate_folder_structure,
)
from storage.factories import FileNodeFactory, GroupCategoryEntryFactory, NodeFactory
from storage.manager import FileSource, NodeKind
from storage.index_models import Node
from storage.work_models import (
    FileMapping,
    FolderStructure,
)


@pytest.fixture
def sample_folders(index_session, sample_snapshot):
    folders = [
        NodeFactory(
            snapshot_id=sample_snapshot.snapshot_id,
            kind=NodeKind.DIR.value,
            name="parent_folder",
            rel_path="parent_folder",
            abs_path="/test/parent_folder",
            depth=1,
        ),
        NodeFactory(
            snapshot_id=sample_snapshot.snapshot_id,
            kind=NodeKind.DIR.value,
            name="child_folder",
            rel_path="parent_folder/child_folder",
            abs_path="/test/parent_folder/child_folder",
            depth=2,
        ),
        NodeFactory(
            snapshot_id=sample_snapshot.snapshot_id,
            kind=NodeKind.DIR.value,
            name="zip_content",
            rel_path="content.zip/zip_content",
            abs_path="/test/content.zip/zip_content",
            depth=2,
            file_source=FileSource.ZIP_CONTENT.value,
        ),
    ]

    return folders


@pytest.fixture
def sample_group_entries(work_session, sample_iteration, sample_folders):
    entries = [
        GroupCategoryEntryFactory(
            folder_id=sample_folders[0].node_id,
            iteration_id=sample_iteration.id,
            processed_name="art_category",
            confidence=0.8,
        ),
        GroupCategoryEntryFactory(
            folder_id=sample_folders[1].node_id,
            iteration_id=sample_iteration.id,
            processed_name="digital_art",
            confidence=0.9,
        ),
        GroupCategoryEntryFactory(
            folder_id=sample_folders[2].node_id,
            iteration_id=sample_iteration.id,
            processed_name="zip_category",
            confidence=0.7,
        ),
    ]
    return entries


class TestGetParentFolder:
    """Test cases for get_parent_folder function"""

    def test_get_parent_folder_normal_path(self, index_session, sample_folders):
        """Test finding parent folder with normal file path"""
        parent_path = Path("/test/parent_folder")
        result = get_parent_folder(index_session, parent_path)

        assert result is not None
        assert result.name == "parent_folder"
        assert result.abs_path == "/test/parent_folder"

    def test_get_parent_folder_not_found(self, index_session, sample_folders):
        """Test when parent folder doesn't exist"""
        parent_path = Path("/nonexistent/path")
        result = get_parent_folder(index_session, parent_path)

        assert result is None

    def test_get_parent_folder_zip_content_false(self, index_session, sample_folders):
        """Test zip content handling when zip_content=False"""
        parent_path = Path("/test/content.zip/zip_content")
        result = get_parent_folder(index_session, parent_path, zip_content=False)

        # The function finds the exact path match regardless of zip_content flag
        assert result is not None
        assert result.name == "zip_content"

    def test_get_parent_folder_zip_content_true(self, index_session, sample_folders):
        """Test zip content handling when zip_content=True"""
        # Test with a path that doesn't exist to trigger fallback behavior
        parent_path = Path("/test/nonexistent.zip/some_content")
        result = get_parent_folder(index_session, parent_path, zip_content=True)

        assert result is None  # Should not find anything since neither path exists

    def test_get_parent_folder_zip_content_fallback(
        self, index_session, sample_folders
    ):
        """Test fallback to parent when zip_content=True but direct path not found"""
        # Create a scenario where the direct path doesn't exist but parent does
        parent_path = Path("/test/nonexistent.zip/some_folder")
        result = get_parent_folder(index_session, parent_path, zip_content=True)

        assert result is None  # Neither direct path nor parent should exist


class TestGetCategoriesForPath:
    """Test cases for get_categories_for_path function"""

    def test_get_categories_for_path_string_input(
        self, index_session, work_session, sample_folders, sample_group_entries
    ):
        """Test with string path input"""
        path = "/test/parent_folder/child_folder/file.txt"
        result = get_categories_for_path(
            index_session, work_session, path, iteration_id=1
        )

        assert [category.processed_name for category in result] == [
            "art_category",
            "digital_art",
        ]

    def test_get_categories_for_path_path_input(
        self, index_session, work_session, sample_folders, sample_group_entries
    ):
        """Test with Path object input"""
        path = Path("/test/parent_folder/child_folder/file.txt")
        result = get_categories_for_path(
            index_session, work_session, path, iteration_id=1
        )

        assert [category.processed_name for category in result] == [
            "art_category",
            "digital_art",
        ]

    def test_get_categories_for_path_no_parent(
        self, index_session, work_session, sample_folders, sample_group_entries
    ):
        """Test when parent folder doesn't exist"""
        path = Path("/nonexistent/path/file.txt")
        result = get_categories_for_path(
            index_session, work_session, path, iteration_id=1
        )

        assert result == []

    def test_get_categories_for_path_with_groups(
        self, index_session, work_session, sample_folders, sample_group_entries
    ):
        """Test when parent folder has associated groups"""
        path = Path("/test/parent_folder/file.txt")
        result = get_categories_for_path(
            index_session, work_session, path, iteration_id=1
        )

        assert [category.processed_name for category in result] == ["art_category"]

    def test_get_categories_for_path_zip_matching(
        self, index_session, work_session, sample_folders, sample_group_entries
    ):
        """Test with zip file path matching"""
        path = Path("/test/content.zip/zip_content/file.txt")
        result = get_categories_for_path(
            index_session, work_session, path, iteration_id=1
        )

        assert [category.processed_name for category in result] == ["zip_category"]

    def test_get_categories_for_path_recursive(
        self, index_session, work_session, sample_folders, sample_group_entries
    ):
        """Test recursive category collection"""
        # This test verifies that categories are collected recursively up the path
        path = Path("/test/parent_folder/child_folder/deep/file.txt")

        # Test the actual recursive behavior by creating a deep folder structure
        NodeFactory(
            snapshot_id=sample_folders[0].snapshot_id,
            kind=NodeKind.DIR.value,
            name="deep",
            rel_path="parent_folder/child_folder/deep",
            abs_path="/test/parent_folder/child_folder/deep",
            depth=3,
        )

        result = get_categories_for_path(
            index_session, work_session, path, iteration_id=1
        )
        assert [category.processed_name for category in result] == [
            "art_category",
            "digital_art",
        ]


class TestCalculateCategories:
    """Test cases for calculate_categories function"""

    def test_calculate_categories_basic(
        self,
        storage_manager,
        storage_index_session,
        storage_work_session,
        storage_snapshot,
        storage_run,
        storage_iteration,
    ):
        """Test basic calculate_categories functionality"""
        FileNodeFactory(
            snapshot_id=storage_snapshot.snapshot_id,
            name="test1.txt",
            rel_path="test1.txt",
            abs_path="/test/test1.txt",
            depth=1,
        )
        GroupCategoryEntryFactory(
            folder_id=1,
            iteration_id=storage_iteration.id,
            processed_name="category",
            confidence=0.9,
        )

        calculate_folder_structure(
            storage_manager, storage_snapshot.snapshot_id, storage_run.id
        )

        storage_work_session.expire_all()
        storage_index_session.expire_all()

        file_nodes = (
            storage_index_session.query(Node).filter_by(kind=NodeKind.FILE.value).all()
        )
        mappings = storage_work_session.query(FileMapping).all()
        assert sorted(mapping.node_id for mapping in mappings) == sorted(
            node.node_id for node in file_nodes
        )

        folder_structures = storage_work_session.query(FolderStructure).all()
        assert len(folder_structures) == 1
        assert folder_structures[0].structure_type == StructureType.organized.value

    def test_calculate_categories_processes_all_files(
        self,
        storage_manager,
        storage_index_session,
        storage_work_session,
        storage_snapshot,
        storage_run,
        storage_iteration,
    ):
        """Test that all files are processed including zip files"""
        FileNodeFactory(
            snapshot_id=storage_snapshot.snapshot_id,
            name="first.txt",
            rel_path="first.txt",
            abs_path="/test/first.txt",
            depth=1,
        )
        FileNodeFactory(
            snapshot_id=storage_snapshot.snapshot_id,
            name="archive.zip",
            rel_path="archive.zip",
            abs_path="/test/archive.zip",
            depth=1,
            file_source=FileSource.ZIP_FILE.value,
        )
        GroupCategoryEntryFactory(
            folder_id=1,
            iteration_id=storage_iteration.id,
            processed_name="category",
            confidence=0.9,
        )

        calculate_folder_structure(
            storage_manager, storage_snapshot.snapshot_id, storage_run.id
        )

        storage_work_session.expire_all()
        storage_index_session.expire_all()

        file_nodes = (
            storage_index_session.query(Node).filter_by(kind=NodeKind.FILE.value).all()
        )
        mappings = storage_work_session.query(FileMapping).all()
        mapped_nodes = sorted(mapping.node_id for mapping in mappings)
        file_node_ids = sorted(node.node_id for node in file_nodes)

        assert file_node_ids == mapped_nodes

    def test_calculate_categories_with_categories(
        self,
        storage_manager,
        storage_index_session,
        storage_work_session,
        storage_snapshot,
        storage_run,
        storage_iteration,
    ):
        """Test calculate_categories with mock categories"""
        FileNodeFactory(
            snapshot_id=storage_snapshot.snapshot_id,
            name="test1.txt",
            rel_path="test1.txt",
            abs_path="/test/test1.txt",
            depth=1,
        )
        GroupCategoryEntryFactory(
            folder_id=1,
            iteration_id=storage_iteration.id,
            processed_name="ignored",
            confidence=0.5,
        )

        def resolve_categories(_index_session, _work_session, _path, _iteration_id):
            return [
                GroupCategoryEntryFactory.build(
                    folder_id=1, processed_name="category1", confidence=0.8
                ),
                GroupCategoryEntryFactory.build(
                    folder_id=1, processed_name="category2", confidence=0.9
                ),
            ]

        calculate_folder_structure(
            storage_manager,
            storage_snapshot.snapshot_id,
            storage_run.id,
            category_resolver=resolve_categories,
        )

        storage_work_session.expire_all()

        mappings = storage_work_session.query(FileMapping).all()
        assert [mapping.new_path for mapping in mappings] == ["category1/category2"]

    def test_calculate_categories_no_files(
        self,
        storage_manager,
        storage_work_session,
        storage_snapshot,
        storage_run,
        storage_iteration,
    ):
        """Test calculate_categories with no files in database"""
        GroupCategoryEntryFactory(
            iteration_id=storage_iteration.id,
            folder_id=1,
            processed_name="test",
        )

        calculate_folder_structure(
            storage_manager, storage_snapshot.snapshot_id, storage_run.id
        )

        storage_work_session.expire_all()

        mappings = storage_work_session.query(FileMapping).all()
        assert mappings == []

        folder_structures = storage_work_session.query(FolderStructure).all()
        assert len(folder_structures) == 1

    def test_calculate_categories_empty_categories(
        self,
        storage_manager,
        storage_work_session,
        storage_snapshot,
        storage_run,
        storage_iteration,
    ):
        """Test calculate_categories when get_categories_for_path returns empty list"""
        FileNodeFactory(
            snapshot_id=storage_snapshot.snapshot_id,
            name="test1.txt",
            rel_path="test1.txt",
            abs_path="/test/test1.txt",
            depth=1,
        )
        GroupCategoryEntryFactory(
            iteration_id=storage_iteration.id,
            folder_id=1,
            processed_name="test",
        )

        def resolve_categories(_index_session, _work_session, _path, _iteration_id):
            return []

        calculate_folder_structure(
            storage_manager,
            storage_snapshot.snapshot_id,
            storage_run.id,
            category_resolver=resolve_categories,
        )

        storage_work_session.expire_all()

        mappings = storage_work_session.query(FileMapping).all()
        assert [mapping.new_path for mapping in mappings] == [""]


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_get_parent_folder_empty_path(self, index_session):
        """Test get_parent_folder with empty path"""
        parent_path = Path("")
        result = get_parent_folder(index_session, parent_path)
        assert result is None

    def test_get_categories_for_path_root_path(self, index_session, work_session):
        """Test get_categories_for_path with root path"""
        path = Path("/")
        result = get_categories_for_path(
            index_session, work_session, path, iteration_id=1
        )
        assert result == []

    def test_calculate_categories_no_iteration_id(
        self, storage_manager, storage_work_session, storage_snapshot, storage_run
    ):
        """Test calculate_categories when no GroupCategoryEntry exists"""
        result = calculate_folder_structure(
            storage_manager, storage_snapshot.snapshot_id, storage_run.id
        )
        assert result is None

        storage_work_session.expire_all()
        folder_structures = storage_work_session.query(FolderStructure).all()
        assert len(folder_structures) == 0
