import tempfile
from virtual_folder import VirtualFolder, VirtualFile
from test_utils import build_virtual_fs
from pathlib import Path


class TestVirtualFolder:
    def test_add_virtual_file_base(self):
        """
        add virtual file with no conflicts
        """
        base_structure = {"testfile": "testfile"}
        virtual_fs = build_virtual_fs(base_structure)

        new_virtual_file = VirtualFile(Path("testfile2"))
        virtual_fs.add_virtual_file(new_virtual_file)

        keys = list(virtual_fs.contents.keys())
        assert keys == ["testfile", "testfile2"]

    def test_add_virtual_file_one_conflict(self):
        """
        add virtual files with one conflicting filename
        """
        base_structure = {"testfile.txt": "testfile.txt"}
        virtual_fs = build_virtual_fs(base_structure)

        new_virtual_file = VirtualFile(Path("testfile.txt"))
        virtual_fs.add_virtual_file(new_virtual_file)

        keys = list(virtual_fs.contents.keys())
        assert keys == ["testfile.txt", "testfile-1.txt"]

    def test_add_virtual_file_several_conflicts(self):
        """
        add virtual files where there are already
        multiple conflicting filenames
        """
        base_structure = {"testfile.txt": "testfile.txt"}
        virtual_fs = build_virtual_fs(base_structure)

        new_virtual_file = VirtualFile(Path("testfile.txt"))
        virtual_fs.add_virtual_file(new_virtual_file)
        new_virtual_file = VirtualFile(Path("testfile.txt"))
        virtual_fs.add_virtual_file(new_virtual_file)

        keys = list(virtual_fs.contents.keys())
        assert keys == ["testfile.txt", "testfile-1.txt", "testfile-2.txt"]

    def test_merge_subfolders_base(self):
        """
        uncomplicated merge - no nesting, no conflicts
        """
        base_structure_1 = {"test": {"test.txt": "test.txt"}}
        virtual_fs = build_virtual_fs(base_structure_1)

        base_structure_2 = {"test": {"test2.txt": "test2.txt"}}
        virtual_merge = build_virtual_fs(base_structure_2).contents["test"]

        virtual_fs.merge_subfolders(virtual_merge)
        output_structure_base = virtual_fs.get_folders_dict()
        expected_base = {"root": {"test": {"test.txt": "", "test2.txt": ""}}}

        assert expected_base == output_structure_base

        output_structure_merged = virtual_merge.get_folders_dict()
        expected_merge = {"test": {}}
        assert expected_merge == output_structure_merged

    def test_merge_subfolders_nested(self):
        """
        Nested subfolders with no conflicts
        """
        base_structure_1 = {"test": {"test1": {"test.txt": "test.txt"}}}
        virtual_fs = build_virtual_fs(base_structure_1)

        base_structure_2 = {"test": {"test1": {"test2.txt": "test2.txt"}}}
        virtual_merge = build_virtual_fs(base_structure_2).contents["test"]

        virtual_fs.merge_subfolders(virtual_merge)
        output_structure_base = virtual_fs.get_folders_dict()
        expected_base = {"root": {"test": {"test1": {"test.txt": "", "test2.txt": ""}}}}

        assert expected_base == output_structure_base

        output_structure_merged = virtual_merge.get_folders_dict()
        expected_merge = {"test": {}}
        assert expected_merge == output_structure_merged

    def test_merge_subfolders_matching_files(self):
        """
        Nested subfolders with matching files
        (ie. same filename and same file instance)
        """
        with tempfile.NamedTemporaryFile() as tf:
            base_structure_1 = {"test": {"test.txt": tf.name}}
            virtual_fs = build_virtual_fs(base_structure_1)

            base_structure_2 = {"test": {"test.txt": tf.name}}
            virtual_merge = build_virtual_fs(base_structure_2).contents["test"]

            virtual_fs.merge_subfolders(virtual_merge)
            output_structure_base = virtual_fs.get_folders_dict()
            expected_base = {"root": {"test": {"test.txt": ""}}}

            assert expected_base == output_structure_base

            output_structure_merged = virtual_merge.get_folders_dict()
            expected_merge = {"test":  {"test.txt": ""}}
            assert expected_merge == output_structure_merged

    def test_merge_subfolders_matching_filenames(self):
        """
        Nested subfolders with matching filenames
        but different files
        """
        with tempfile.NamedTemporaryFile() as tf, tempfile.NamedTemporaryFile() as tf2:
            base_structure_1 = {"test": {"test.txt": tf.name}}
            virtual_fs = build_virtual_fs(base_structure_1)
            tf.write(b"test")
            tf.flush()


            base_structure_2 = {"test": {"test.txt": tf2.name}}
            virtual_merge = build_virtual_fs(base_structure_2).contents["test"]

            virtual_fs.merge_subfolders(virtual_merge)
            output_structure_base = virtual_fs.get_folders_dict()
            expected_base = {"root": {"test": {"test.txt": "", "test-1.txt": ""}}}

            assert expected_base == output_structure_base

            output_structure_merged = virtual_merge.get_folders_dict()
            expected_merge = {"test":  {}}
            assert expected_merge == output_structure_merged

    def test_merge_subfolders_complex(self):
        """
        Nested subfolders with matching files and
        matching filenames
        """
        with tempfile.NamedTemporaryFile() as tf, tempfile.NamedTemporaryFile() as tf2:
            base_structure_1 = {"test": { "test1" : {"test.txt": tf.name}}}
            virtual_fs = build_virtual_fs(base_structure_1)
            tf.write(b"test")
            tf.flush()

            base_structure_2 = {"test": {"test1" : {"test.txt": tf2.name}}}
            virtual_merge = build_virtual_fs(base_structure_2).contents["test"]

            virtual_fs.merge_subfolders(virtual_merge)
            output_structure_base = virtual_fs.get_folders_dict()
            expected_base = {"root": {"test":{ "test1" : {"test.txt": "", "test-1.txt": ""}}}}

            assert expected_base == output_structure_base

            output_structure_merged = virtual_merge.get_folders_dict()
            expected_merge = {"test":  {}}
            assert expected_merge == output_structure_merged
