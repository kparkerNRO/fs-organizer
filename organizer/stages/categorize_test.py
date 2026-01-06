import pytest
from api.api import StructureType
from storage.factories import FileNodeFactory, GroupCategoryEntryFactory, NodeFactory
from storage.index_models import Node
from storage.manager import FileSource, NodeKind
from storage.work_models import (
    FileMapping,
    FolderStructure,
)
from utils.folder_structure import calculate_folder_structure_for_categories

from stages.categorize import (
    get_categories_for_node,
)


@pytest.fixture
def sample_folders(index_session, sample_snapshot):
    parent_folder = NodeFactory(
        snapshot_id=sample_snapshot.id,
        kind=NodeKind.DIR.value,
        name="parent_folder",
        rel_path="parent_folder",
        abs_path="/test/parent_folder",
        depth=1,
    )
    # Flush to ensure parent_folder is committed before child references it
    index_session.flush()

    child_folder = NodeFactory(
        snapshot_id=sample_snapshot.id,
        kind=NodeKind.DIR.value,
        name="child_folder",
        rel_path="parent_folder/child_folder",
        abs_path="/test/parent_folder/child_folder",
        depth=2,
        parent_node_id=parent_folder.id,
    )

    zip_content = NodeFactory(
        snapshot_id=sample_snapshot.id,
        kind=NodeKind.DIR.value,
        name="zip_content",
        rel_path="content.zip/zip_content",
        abs_path="/test/content.zip/zip_content",
        depth=2,
        file_source=FileSource.ZIP_CONTENT.value,
    )

    return [parent_folder, child_folder, zip_content]


@pytest.fixture
def sample_group_entries(work_session, sample_iteration, sample_folders):
    entries = [
        GroupCategoryEntryFactory(
            folder_id=sample_folders[0].id,
            iteration=sample_iteration,
            processed_name="art_category",
            confidence=0.8,
        ),
        GroupCategoryEntryFactory(
            folder_id=sample_folders[1].id,
            iteration=sample_iteration,
            processed_name="digital_art",
            confidence=0.9,
        ),
        GroupCategoryEntryFactory(
            folder_id=sample_folders[2].id,
            iteration=sample_iteration,
            processed_name="zip_category",
            confidence=0.7,
        ),
    ]
    return entries


class TestGetCategoriesForPath:
    """Test cases for get_categories_for_path function"""

    def test_get_categories_for_path_no_parent(
        self,
        index_session,
        work_session,
        sample_folders,
        sample_group_entries,
        sample_iteration,
        sample_snapshot,
    ):
        """Test when parent folder doesn't exist"""
        test_node = NodeFactory(
            snapshot_id=sample_snapshot.id,
            kind=NodeKind.FILE,
        )
        result = get_categories_for_node(
            index_session,
            work_session,
            test_node,
            iteration_id=sample_iteration.id,
        )

        assert result == []

    def test_get_categories_for_path_with_groups(
        self,
        index_session,
        work_session,
        sample_folders,
        sample_group_entries,
        sample_iteration,
        sample_snapshot,
    ):
        """Test when parent folder has associated groups"""
        test_node = NodeFactory(
            snapshot_id=sample_snapshot.id,
            kind=NodeKind.FILE,
            parent_node_id=sample_folders[1].id,
        )
        result = get_categories_for_node(
            index_session,
            work_session,
            test_node,
            iteration_id=sample_iteration.id,
        )

        assert [category.processed_name for category in result] == ["art_category", "digital_art"]

    def test_get_categories_for_path_zip_matching(
        self,
        index_session,
        work_session,
        sample_folders,
        sample_group_entries,
        sample_iteration,
        sample_snapshot,
    ):
        """Test with zip file path matching"""
        test_node = NodeFactory(
            snapshot_id=sample_snapshot.id,
            kind=NodeKind.FILE,
            parent_node_id=sample_folders[2].id,
        )
        result = get_categories_for_node(
            index_session,
            work_session,
            test_node,
            iteration_id=sample_iteration.id,
        )

        assert [category.processed_name for category in result] == ["zip_category"]

    @pytest.mark.skip("Need to redesign test for id-based search")
    def test_get_categories_for_path_recursive(
        self,
        index_session,
        work_session,
        sample_folders,
        sample_group_entries,
        sample_iteration,
    ):
        """Test recursive category collection"""
        # This test verifies that categories are collected recursively up the path

        # Test the actual recursive behavior by creating a deep folder structure
        NodeFactory(
            snapshot_id=sample_folders[0].snapshot_id,
            kind=NodeKind.DIR.value,
            name="deep",
            rel_path="parent_folder/child_folder/deep",
            abs_path="/test/parent_folder/child_folder/deep",
            depth=3,
        )
        index_session.commit()

        test_node = NodeFactory(
            snapshot_id=sample_iteration.snapshot_id,
            kind=NodeKind.FILE,
            abs_path="/nonexistent/path/file.txt",
        )

        result = get_categories_for_node(
            index_session,
            work_session,
            test_node,
            iteration_id=sample_iteration.id,
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
            snapshot_id=storage_snapshot.id,
            name="test1.txt",
            rel_path="test1.txt",
            abs_path="/test/test1.txt",
            depth=1,
        )
        GroupCategoryEntryFactory(
            folder_id=1,
            iteration=storage_iteration,
            processed_name="category",
            confidence=0.9,
        )

        calculate_folder_structure_for_categories(
            storage_manager, storage_snapshot.id, storage_run.id
        )

        storage_work_session.expire_all()
        storage_index_session.expire_all()

        file_nodes = (
            storage_index_session.query(Node).filter_by(kind=NodeKind.FILE.value).all()
        )
        mappings = storage_work_session.query(FileMapping).all()
        assert sorted(mapping.node_id for mapping in mappings) == sorted(
            node.id for node in file_nodes
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
            snapshot_id=storage_snapshot.id,
            name="first.txt",
            rel_path="first.txt",
            abs_path="/test/first.txt",
            depth=1,
        )
        FileNodeFactory(
            snapshot_id=storage_snapshot.id,
            name="archive.zip",
            rel_path="archive.zip",
            abs_path="/test/archive.zip",
            depth=1,
            file_source=FileSource.ZIP_FILE.value,
        )
        GroupCategoryEntryFactory(
            folder_id=1,
            iteration=storage_iteration,
            processed_name="category",
            confidence=0.9,
        )

        calculate_folder_structure_for_categories(
            storage_manager, storage_snapshot.id, storage_run.id
        )

        storage_work_session.expire_all()
        storage_index_session.expire_all()

        file_nodes = (
            storage_index_session.query(Node).filter_by(kind=NodeKind.FILE.value).all()
        )
        mappings = storage_work_session.query(FileMapping).all()
        mapped_nodes = sorted(mapping.node_id for mapping in mappings)
        file_node_ids = sorted(node.id for node in file_nodes)

        assert file_node_ids == mapped_nodes

    # @pytest.mark.skip("Dependency injection on calculate_folder_structure")
    # def test_calculate_categories_with_categories(
    #     self,
    #     storage_manager,
    #     storage_index_session,
    #     storage_work_session,
    #     storage_snapshot,
    #     storage_run,
    #     storage_iteration,
    # ):
    #     """Test calculate_categories with mock categories"""
    #     FileNodeFactory(
    #         snapshot_id=storage_snapshot.id,
    #         name="test1.txt",
    #         rel_path="test1.txt",
    #         abs_path="/test/test1.txt",
    #         depth=1,
    #     )
    #     GroupCategoryEntryFactory(
    #         folder_id=1,
    #         iteration=storage_iteration,
    #         processed_name="ignored",
    #         confidence=0.5,
    #     )

    #     def resolve_categories(_index_session, _work_session, _path, _iteration_id):
    #         return [
    #             GroupCategoryEntryFactory.build(
    #                 folder_id=1, processed_name="category1", confidence=0.8
    #             ),
    #             GroupCategoryEntryFactory.build(
    #                 folder_id=1, processed_name="category2", confidence=0.9
    #             ),
    #         ]

    #     calculate_folder_structure_for_categories(
    #         storage_manager,
    #         storage_snapshot.id,
    #         storage_run.id,
    #         category_resolver=resolve_categories,
    #     )

    #     storage_work_session.expire_all()

    #     mappings = storage_work_session.query(FileMapping).all()
    #     assert [mapping.new_path for mapping in mappings] == ["category1/category2"]

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
            iteration=storage_iteration,
            folder_id=1,
            processed_name="test",
        )

        calculate_folder_structure_for_categories(
            storage_manager, storage_snapshot.id, storage_run.id
        )

        storage_work_session.expire_all()

        mappings = storage_work_session.query(FileMapping).all()
        assert mappings == []

        folder_structures = storage_work_session.query(FolderStructure).all()
        assert len(folder_structures) == 1

    # @pytest.mark.skip("Dependency injection on calculate_folder_structure")
    # def test_calculate_categories_empty_categories(
    #     self,
    #     storage_manager,
    #     storage_work_session,
    #     storage_snapshot,
    #     storage_run,
    #     storage_iteration,
    # ):
    #     """Test calculate_categories when get_categories_for_path returns empty list"""
    #     FileNodeFactory(
    #         snapshot_id=storage_snapshot.id,
    #         name="test1.txt",
    #         rel_path="test1.txt",
    #         abs_path="/test/test1.txt",
    #         depth=1,
    #     )
    #     GroupCategoryEntryFactory(
    #         iteration=storage_iteration,
    #         folder_id=1,
    #         processed_name="test",
    #     )

    #     def resolve_categories(_index_session, _work_session, _path, _iteration_id):
    #         return []

    #     calculate_folder_structure_for_categories(
    #         storage_manager,
    #         storage_snapshot.id,
    #         storage_run.id,
    #         category_resolver=resolve_categories,
    #     )

    #     storage_work_session.expire_all()

    #     mappings = storage_work_session.query(FileMapping).all()
    #     assert [mapping.new_path for mapping in mappings] == [""]


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_calculate_categories_no_iteration_id(
        self, storage_manager, storage_work_session, storage_snapshot, storage_run
    ):
        """Test calculate_categories when no GroupCategoryEntry exists"""
        result = calculate_folder_structure_for_categories(
            storage_manager, storage_snapshot.id, storage_run.id
        )
        assert result is None

        storage_work_session.expire_all()
        folder_structures = storage_work_session.query(FolderStructure).all()
        assert len(folder_structures) == 0
