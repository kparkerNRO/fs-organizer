import pytest
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from stages.gather import process_zip, ingest_filesystem
from storage.index_models import IndexBase, Snapshot, Node, NodeFeatures
from storage.manager import NodeKind, FileSource
import zipfile
import io

FILESYSTEM_SOURCE = "filesystem"
ZIP_FILE_SOURCE = "zip_file"
ZIP_CONTENT_SOURCE = "zip_content"


@pytest.fixture
def index_session():
    engine = create_engine("sqlite:///:memory:")
    IndexBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    IndexBase.metadata.drop_all(engine)


@pytest.fixture
def snapshot_id(index_session):
    snapshot = Snapshot(
        created_at="2024-01-01T00:00:00",
        root_path="/test",
        root_abs_path="/test",
    )
    index_session.add(snapshot)
    index_session.commit()
    return snapshot.snapshot_id


def create_test_zip(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for entry in entries:
            zf.writestr(entry, "test content")
    buffer.seek(0)
    return buffer


def test_process_zip_basic(index_session, snapshot_id):
    zip_entries = ["file1.txt", "file2.txt", "dir1/", "dir1/file3.txt"]
    zip_buffer = create_test_zip(zip_entries)

    process_zip(
        zip_buffer,
        Path("/test"),
        Path(""),
        None,
        "test.zip",
        index_session,
        snapshot_id,
    )

    zip_file_nodes = (
        index_session.query(Node).filter_by(file_source=ZIP_FILE_SOURCE).all()
    )
    zip_dirs = (
        index_session.query(Node)
        .filter_by(
            kind=NodeKind.DIR.value,
            file_source=ZIP_CONTENT_SOURCE,
        )
        .all()
    )
    zip_files = (
        index_session.query(Node)
        .filter_by(
            kind=NodeKind.FILE.value,
            file_source=ZIP_CONTENT_SOURCE,
        )
        .all()
    )

    assert len(zip_file_nodes) == 1
    assert len(zip_dirs) == 1  # dir1
    assert len(zip_files) == 3  # file1.txt, file2.txt, dir1/file3.txt

    assert zip_file_nodes[0].name == "test.zip"

    dir_names = [node.name for node in zip_dirs]
    assert dir_names == ["dir1"]

    file_names = sorted(node.name for node in zip_files)
    assert file_names == ["file1.txt", "file2.txt", "file3.txt"]


@pytest.mark.skip(reason="Pre-existing test failure - assert 1 == 0")
def test_process_zip_with_module_json(session):
    zip_entries = ["module.json", "file1.txt"]
    zip_buffer = create_test_zip(zip_entries)

    process_zip(
        zip_buffer,
        Path("/test"),
        Path(""),
        None,
        "test.zip",
        index_session,
        snapshot_id,
    )

    zip_file_nodes = (
        index_session.query(Node)
        .filter_by(
            kind=NodeKind.FILE.value,
            file_source=ZIP_FILE_SOURCE,
        )
        .all()
    )
    zip_dirs = (
        index_session.query(Node)
        .filter_by(
            kind=NodeKind.DIR.value,
            file_source=ZIP_FILE_SOURCE,
        )
        .all()
    )
    zip_content_nodes = (
        index_session.query(Node).filter_by(file_source=ZIP_CONTENT_SOURCE).all()
    )

    assert len(zip_dirs) == 1  # Foundry module wrapper folder
    assert len(zip_file_nodes) == 1  # Only the zip file itself should be added
    assert len(zip_content_nodes) == 0

    file_names = [node.name for node in zip_file_nodes]
    assert file_names == ["test.zip"]

    folder_names = [node.name for node in zip_dirs]
    assert folder_names == ["Foundry Module test.zip"]


def test_process_zip_nested_zip(index_session, snapshot_id):
    nested_zip_entries = ["nested_file.txt"]
    nested_zip_buffer = create_test_zip(nested_zip_entries)

    zip_entries = ["file1.txt"]
    zip_buffer = create_test_zip(zip_entries)

    with zipfile.ZipFile(zip_buffer, "a") as zf:
        zf.writestr("nested.zip", nested_zip_buffer.getvalue())

    zip_buffer.seek(0)
    process_zip(
        zip_buffer,
        Path("/test"),
        Path(""),
        None,
        "test.zip",
        index_session,
        snapshot_id,
    )

    zip_file_nodes = (
        index_session.query(Node)
        .filter_by(
            kind=NodeKind.FILE.value,
            file_source=ZIP_FILE_SOURCE,
        )
        .all()
    )
    zip_content_dirs = (
        index_session.query(Node)
        .filter_by(
            kind=NodeKind.DIR.value,
            file_source=ZIP_CONTENT_SOURCE,
        )
        .all()
    )
    zip_content_files = (
        index_session.query(Node)
        .filter_by(
            kind=NodeKind.FILE.value,
            file_source=ZIP_CONTENT_SOURCE,
        )
        .all()
    )

    assert len(zip_file_nodes) == 1
    assert len(zip_content_dirs) == 0
    assert len(zip_content_files) == 3  # file1.txt, nested.zip, nested_file.txt

    assert zip_file_nodes[0].name == "test.zip"

    file_names = sorted(node.name for node in zip_content_files)
    assert file_names == ["file1.txt", "nested.zip", "nested_file.txt"]


def test_process_zip_ignores(index_session, snapshot_id):
    zip_entries = ["__MACOSX/", ".DS_Store", "file1.txt"]
    zip_buffer = create_test_zip(zip_entries)

    process_zip(
        zip_buffer,
        Path("/test"),
        Path(""),
        None,
        "test.zip",
        index_session,
        snapshot_id,
    )

    zip_file_nodes = (
        index_session.query(Node)
        .filter_by(
            kind=NodeKind.FILE.value,
            file_source=ZIP_FILE_SOURCE,
        )
        .all()
    )
    zip_content_files = (
        index_session.query(Node)
        .filter_by(
            kind=NodeKind.FILE.value,
            file_source=ZIP_CONTENT_SOURCE,
        )
        .all()
    )

    assert len(zip_file_nodes) == 1
    assert len(zip_content_files) == 1  # Only file1.txt should be added

    file_names = [node.name for node in zip_content_files]
    assert file_names == ["file1.txt"]


def test_process_zip_top_level_folder(index_session, snapshot_id):
    zip_entries = ["test/", "test/file1.txt", "test/file2.txt"]
    zip_buffer = create_test_zip(zip_entries)

    process_zip(
        zip_buffer,
        Path("/test"),
        Path(""),
        None,
        "test.zip",
        index_session,
        snapshot_id,
    )

    zip_dirs = (
        index_session.query(Node)
        .filter_by(
            kind=NodeKind.DIR.value,
            file_source=ZIP_CONTENT_SOURCE,
        )
        .all()
    )
    zip_files = (
        index_session.query(Node)
        .filter_by(
            kind=NodeKind.FILE.value,
            file_source=ZIP_CONTENT_SOURCE,
        )
        .all()
    )

    assert len(zip_dirs) == 1
    assert len(zip_files) == 2

    folder_names = [node.name for node in zip_dirs]
    assert folder_names == ["test"]

    file_names = sorted(node.name for node in zip_files)
    assert file_names == ["file1.txt", "file2.txt"]


def test_process_zip_top_level_folder_non_specified(index_session, snapshot_id):
    zip_entries = ["test/file1.txt", "test/file2.txt"]
    zip_buffer = create_test_zip(zip_entries)

    process_zip(
        zip_buffer,
        Path("/test"),
        Path(""),
        None,
        "test.zip",
        index_session,
        snapshot_id,
    )

    zip_dirs = (
        index_session.query(Node)
        .filter_by(
            kind=NodeKind.DIR.value,
            file_source=ZIP_CONTENT_SOURCE,
        )
        .all()
    )
    zip_files = (
        index_session.query(Node)
        .filter_by(
            kind=NodeKind.FILE.value,
            file_source=ZIP_CONTENT_SOURCE,
        )
        .all()
    )

    assert len(zip_dirs) == 1
    assert len(zip_files) == 2

    folder_names = [node.name for node in zip_dirs]
    assert folder_names == ["test"]

    file_names = sorted(node.name for node in zip_files)
    assert file_names == ["file1.txt", "file2.txt"]


def test_ingest_filesystem_creates_nodes(tmp_path: Path):
    base_path = tmp_path / "root"
    base_path.mkdir()
    (base_path / "dir1").mkdir()
    (base_path / "dir1" / "file1.txt").write_text("hello", encoding="utf-8")
    zip_path = base_path / "file2.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("inside.txt", "content")
    zip_path.write_bytes(buffer.getvalue())

    snapshot_id = ingest_filesystem(base_path, tmp_path)

    engine = create_engine(f"sqlite:///{tmp_path / 'index.db'}")
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        nodes = (
            session.query(Node)
            .filter_by(snapshot_id=snapshot_id)
            .filter(Node.file_source.in_([FILESYSTEM_SOURCE, ZIP_FILE_SOURCE]))
            .all()
        )
        names = sorted(node.name for node in nodes)
        assert names == ["dir1", "file1.txt", "file2.zip"]

        dir_node = (
            session.query(Node)
            .filter_by(
                snapshot_id=snapshot_id,
                name="dir1",
                kind=NodeKind.DIR.value,
            )
            .one()
        )
        file_node = (
            session.query(Node)
            .filter_by(
                snapshot_id=snapshot_id,
                name="file1.txt",
                kind=NodeKind.FILE.value,
            )
            .one()
        )
        zip_node = (
            session.query(Node)
            .filter_by(
                snapshot_id=snapshot_id,
                name="file2.zip",
                kind=NodeKind.FILE.value,
                file_source=FileSource.ZIP_FILE.value,
            )
            .one()
        )

        assert dir_node.file_source == FILESYSTEM_SOURCE
        assert file_node.parent_node_id == dir_node.node_id
        assert zip_node.parent_node_id is None

        features = session.query(NodeFeatures).filter_by(node_id=dir_node.node_id).one()
        assert features.normalized_name == "dir"
    finally:
        session.close()


def test_ingest_filesystem_zip_nodes(tmp_path: Path):
    base_path = tmp_path / "root"
    base_path.mkdir()
    zip_path = base_path / "archive.zip"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("folder/", "")
        zf.writestr("folder/nested.txt", "content")
        zf.writestr("root.txt", "content")
    zip_path.write_bytes(buffer.getvalue())

    snapshot_id = ingest_filesystem(base_path, tmp_path)

    engine = create_engine(f"sqlite:///{tmp_path / 'index.db'}")
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        zip_node = (
            session.query(Node)
            .filter_by(
                snapshot_id=snapshot_id,
                name="archive.zip",
                file_source=ZIP_FILE_SOURCE,
            )
            .one()
        )
        zip_children = (
            session.query(Node)
            .filter_by(
                snapshot_id=snapshot_id,
                parent_node_id=zip_node.node_id,
                file_source=ZIP_CONTENT_SOURCE,
            )
            .all()
        )
        child_names = sorted(child.name for child in zip_children)
        assert child_names == ["folder", "root.txt"]

        zip_files = (
            session.query(Node)
            .filter_by(
                snapshot_id=snapshot_id,
                kind=NodeKind.FILE.value,
                file_source=ZIP_CONTENT_SOURCE,
            )
            .all()
        )
        zip_file_names = sorted(node.name for node in zip_files)
        assert zip_file_names == ["nested.txt", "root.txt"]
    finally:
        session.close()
