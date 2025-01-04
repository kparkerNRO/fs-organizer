import pytest


@pytest.fixture
def test_file(tmp_path):
    file_path = tmp_path / "testfile.txt"
    file_path.write_text("This is a test file")
    yield file_path
