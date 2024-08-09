import json
import tempfile
import zipfile
from virtual_folder import VirtualFolder, VirtualFile, build_folder_structure
from test_utils import build_virtual_fs
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.tables import Base, Exports, Files, Runs, Tags


def test_add_virtual_file_base():
    """
    add virtual file with no conflicts
    """
    base_structure = {"testfile": "testfile"}
    virtual_fs = build_virtual_fs(base_structure)

    new_virtual_file = VirtualFile(Path("testfile2"))
    virtual_fs.add_virtual_file(new_virtual_file)

    keys = list(virtual_fs.contents.keys())
    assert keys == ["testfile", "testfile2"]


def test_add_virtual_file_one_conflict():
    """
    add virtual files with one conflicting filename
    """
    base_structure = {"testfile.txt": "testfile.txt"}
    virtual_fs = build_virtual_fs(base_structure)

    new_virtual_file = VirtualFile(Path("testfile.txt"))
    virtual_fs.add_virtual_file(new_virtual_file)

    keys = list(virtual_fs.contents.keys())
    assert keys == ["testfile.txt", "testfile-1.txt"]


def test_add_virtual_file_several_conflicts():
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


def test_merge_subfolders_base():
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


def test_merge_subfolders_nested():
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


def test_merge_subfolders_matching_files():
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
        expected_merge = {"test": {"test.txt": ""}}
        assert expected_merge == output_structure_merged


def test_merge_subfolders_matching_filenames():
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
        expected_merge = {"test": {}}
        assert expected_merge == output_structure_merged


def test_merge_subfolders_complex():
    """
    Nested subfolders with matching files and
    matching filenames
    """
    with tempfile.NamedTemporaryFile() as tf, tempfile.NamedTemporaryFile() as tf2:
        base_structure_1 = {"test": {"test1": {"test.txt": tf.name}}}
        virtual_fs = build_virtual_fs(base_structure_1)
        tf.write(b"test")
        tf.flush()

        base_structure_2 = {"test": {"test1": {"test.txt": tf2.name}}}
        virtual_merge = build_virtual_fs(base_structure_2).contents["test"]

        virtual_fs.merge_subfolders(virtual_merge)
        output_structure_base = virtual_fs.get_folders_dict()
        expected_base = {
            "root": {"test": {"test1": {"test.txt": "", "test-1.txt": ""}}}
        }

        assert expected_base == output_structure_base

        output_structure_merged = virtual_merge.get_folders_dict()
        expected_merge = {"test": {}}
        assert expected_merge == output_structure_merged


@pytest.fixture()
def db_session():
    # Create an in-memory SQLite database
    engine = create_engine("sqlite:///:memory:")

    # Create all tables defined in the Base class
    Base.metadata.create_all(engine)

    # Create a session factory
    Session = sessionmaker(bind=engine)

    # Create a session
    session = Session()

    # Yield the session to the test function
    yield session

    # Close the session
    session.close()

    # Drop all tables defined in the Base class
    Base.metadata.drop_all(engine)


def test_write_to_db(db_session, test_file):
    """
    test writing a virtual FS to the database
    """
    base_structure = {
        "tom": {
            "PDF": {
                "sample_folder": {
                    "day": {
                        "base": {"sample_file": test_file.name},
                        "tile": {"sample_file": test_file.name},
                    },
                    "night": {"sample_file": test_file.name},
                },
                "sample_file": test_file.name,
            },
            "VTT": {
                "sample_folder": {
                    "day": {
                        "base": {"sample_file": test_file.name},
                        "tile": {"sample_file": test_file.name},
                    },
                    "night": {"sample_file": test_file.name},
                },
                "sample_file": test_file.name,
            },
            "sample_file": test_file.name,
        }
    }
    virtual_fs = build_virtual_fs(base_structure)

    run = Runs(root_folder="testfile")
    db_session.add(run)
    db_session.commit()

    virtual_fs.create_database_export(db_session, run.id)

    export_record = db_session.query(Exports)
    assert export_record.count() == 1
    assert export_record.first().folder_structure == json.dumps(virtual_fs.get_folders_dict(show_files=False))

    tags = db_session.query(Tags)
    print("tags")
    for tag in tags:
        print(tag.original_name)
    print()

    files = db_session.query(Files)
    print("files")
    for file in files:
        print(file.original_path, file.file_name,)

    assert False

def test_build_folder_structure_with_zipfiles():
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_dir = Path(tmp_dir)

        # Create some files and folders inside the test directory
        file1 = test_dir / "file1.txt"
        file1.touch()
        file2 = test_dir / "file2.txt"
        file2.touch()
        folder1 = test_dir / "folder1"
        folder1.mkdir()
        file3 = folder1 / "file3.txt"
        file3.touch()
        folder2 = folder1 / "folder2"
        folder2.mkdir()
        file4 = folder2 / "file4.txt"
        file4.touch()

        # Create a temporary zip file
        zip_file = test_dir / "test.zip"
        with zipfile.ZipFile(zip_file, "w") as zf:
            zf.write(file1, arcname="file1.txt")
            zf.write(file2, arcname="file2.txt")
            zf.write(file3, arcname="folder1/file3.txt")
            zf.write(file4, arcname="folder1/folder2/file4.txt")

        # Build the virtual folder structure
        root_folder = VirtualFolder(path=test_dir)
        build_folder_structure(test_dir, folder=root_folder)

        expected_struct = {
            test_dir.name: {
                "file1.txt": "file1.txt",
                "file2.txt": "file2.txt",
                "folder1": {
                    "file3.txt": "file3.txt",
                    "folder2": {"file4.txt": "file4.txt"},
                },
                "test.zip": {
                    "file1.txt": "file1.txt",
                    "file2.txt": "file2.txt",
                    "folder1": {
                        "file3.txt": "file3.txt",
                        "folder2": {"file4.txt": "file4.txt"},
                    },
                },
            }
        }

        output_struct = root_folder.get_folders_dict()

        assert output_struct == expected_struct
