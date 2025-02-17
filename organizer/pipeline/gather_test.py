import pytest
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pipeline.gather import process_zip
from data_models.database import Base, Folder, File
import zipfile
import io


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


def create_test_zip(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for entry in entries:
            zf.writestr(entry, "test content")
    buffer.seek(0)
    return buffer


def test_process_zip_basic(session):
    zip_entries = ["file1.txt", "file2.txt", "dir1/", "dir1/file3.txt"]
    zip_buffer = create_test_zip(zip_entries)

    process_zip(zip_buffer, Path("/test"), "test.zip", 0, session)

    folders = session.query(Folder).all()
    files = session.query(File).all()

    assert len(folders) == 2  # root folder and dir1
    assert len(files) == 3  # file1.txt, file2.txt, dir1/file3.txt

    folder_names = [folder.folder_name for folder in folders]
    assert folder_names == ["test.zip", "dir1"]

    file_names = [file.file_name for file in files]
    assert file_names == ["file1.txt", "file2.txt", "file3.txt"]


def test_process_zip_with_module_json(session):
    zip_entries = ["module.json", "file1.txt"]
    zip_buffer = create_test_zip(zip_entries)

    process_zip(zip_buffer, Path("/test"), "test.zip", 0, session)

    folders = session.query(Folder).all()
    files = session.query(File).all()

    assert len(folders) == 0  # No folders should be created
    assert len(files) == 1  # Only the zip file itself should be added

    file_names = [file.file_name for file in files]
    assert file_names == ["test.zip"]


def test_process_zip_nested_zip(session):
    nested_zip_entries = ["nested_file.txt"]
    nested_zip_buffer = create_test_zip(nested_zip_entries)

    zip_entries = ["file1.txt"]
    zip_buffer = create_test_zip(zip_entries)

    with zipfile.ZipFile(zip_buffer, "a") as zf:
        zf.writestr("nested.zip", nested_zip_buffer.getvalue())

    zip_buffer.seek(0)
    process_zip(zip_buffer, Path("/test"), "test.zip", 0, session)

    folders = session.query(Folder).all()
    files = session.query(File).all()

    assert len(folders) == 2  # root folder
    assert len(files) == 2  # file1.txt, nested.zip, nested_file.txt

    folder_names = [folder.folder_name for folder in folders]
    assert folder_names == ["test.zip", "nested.zip"]

    file_names = [file.file_name for file in files]
    assert file_names == ["file1.txt", "nested_file.txt"]


def test_process_zip_ignores(session):
    zip_entries = ["__MACOSX/", ".DS_Store", "file1.txt"]
    zip_buffer = create_test_zip(zip_entries)

    process_zip(zip_buffer, Path("/test"), "test.zip", 0, session)

    folders = session.query(Folder).all()
    files = session.query(File).all()

    assert len(folders) == 1  # No folders should be created
    assert len(files) == 1  # Only file1.txt should be added

    folder_names = [folder.folder_name for folder in folders]
    assert folder_names == ["test.zip"]

    file_names = [file.file_name for file in files]
    assert file_names == ["file1.txt"]


def test_process_zip_top_level_folder(session):
    zip_entries = ["test/", "test/file1.txt", "test/file2.txt"]
    zip_buffer = create_test_zip(zip_entries)

    process_zip(zip_buffer, Path("/test"), "test.zip", 0, session)

    folders = session.query(Folder).all()
    files = session.query(File).all()

    assert len(folders) == 1  # root folder
    assert len(files) == 2

    folder_names = [folder.folder_name for folder in folders]
    assert folder_names == ["test"]

    file_names = [file.file_name for file in files]
    assert file_names == ["file1.txt", "file2.txt"]


def test_process_zip_top_level_folder_non_specified(session):
    zip_entries = ["test/file1.txt", "test/file2.txt"]
    zip_buffer = create_test_zip(zip_entries)

    process_zip(zip_buffer, Path("/test"), "test.zip", 0, session)

    folders = session.query(Folder).all()
    files = session.query(File).all()

    assert len(folders) == 1  # root folder
    assert len(files) == 2

    folder_names = [folder.folder_name for folder in folders]
    assert folder_names == ["test"]

    file_names = [file.file_name for file in files]
    assert file_names == ["file1.txt", "file2.txt"]
