from tempfile import TemporaryDirectory, TemporaryFile
from zipfile import ZipFile
from pathlib import Path
from common import ZipBackupState, FileMoveException
import extract_zip
import os
import pytest


class TestExtractZip:
    def setup_method(self):
        self.tmpdir = TemporaryDirectory()
        test_zip_path = Path(self.tmpdir.name, "test.zip")
        with ZipFile(test_zip_path, "w") as zf:
            zf.writestr("test.txt", "Hello world")
        self.test_zip = test_zip_path

    def teardown_method(self):
        try:
            self.tmpdir.cleanup()
        except:
            pass

    def test_extract_zip(self):
        extract_zip.extract_zip_files(Path(self.tmpdir.name), should_execute=True)
        outpath = Path(self.tmpdir.name, "test", "test.txt")
        assert outpath.exists()

    def test_extract_zip_out_dir(self):
        with TemporaryDirectory() as td:
            td_path = Path(td, "test2")
            extract_zip.extract_zip_files(
                Path(self.tmpdir.name), out_dir=td_path, should_execute=True
            )
            outpath = Path(td_path, "test", "test.txt")
            assert outpath.exists()
            assert self.test_zip.exists()

    def test_extract_delete_zip(self):
        extract_zip.extract_zip_files(
            Path(self.tmpdir.name),
            should_execute=True,
            zip_backup_state=ZipBackupState.DELETE,
        )
        outpath = Path(self.test_zip.parent, "test", "test.txt")
        assert outpath.exists()
        assert not self.test_zip.exists()

    def test_extract_move_zip(self):
        with TemporaryDirectory() as td:
            td_path = Path(td, "test2")
            extract_zip.extract_zip_files(
                Path(self.tmpdir.name),
                zip_backup_dir=str(td_path),
                should_execute=True,
                zip_backup_state=ZipBackupState.MOVE,
            )
            outpath = Path(td_path, "test.zip")
            file_path = Path(self.test_zip.parent, "test", "test.txt")
            assert outpath.exists()
            assert file_path.exists()
            assert not self.test_zip.exists()

    def test_extract_move_no_output_folder(self):
        with pytest.raises(FileMoveException):
            extract_zip.extract_zip_files(
                Path(self.tmpdir.name),
                should_execute=True,
                zip_backup_state=ZipBackupState.MOVE,
            )

    
    def test_preserve_modules(self):
        module_zip_path = Path(self.tmpdir.name, "module.zip")
        with ZipFile(module_zip_path, "w") as zf:
            zf.writestr("module.json", "Hello world")

        with TemporaryDirectory() as td:
            td_path = Path(td, "test2")
            extract_zip.extract_zip_files(
                Path(self.tmpdir.name), 
                out_dir=td_path, 
                should_execute=True,
            )
            outpath = Path(td_path, "test", "test.txt")
            assert outpath.exists()
            assert self.test_zip.exists()

            zip_path = Path(td_path, "modules", "module.zip")
            assert zip_path.exists()
            assert module_zip_path.exists()

    def test_preserve_modules_module_path(self):
        module_zip_path = Path(self.tmpdir.name, "module.zip")
        with ZipFile(module_zip_path, "w") as zf:
            zf.writestr("module.json", "Hello world")

        with TemporaryDirectory() as td:
            td_path = Path(td, "test2")
            extract_zip.extract_zip_files(
                Path(self.tmpdir.name), 
                out_dir=td_path, 
                should_execute=True,
                module_dir=td_path
            )
            outpath = Path(td_path, "test", "test.txt")
            assert outpath.exists()
            assert self.test_zip.exists()

            zip_path = Path(td_path, "module.zip")
            assert zip_path.exists()
            assert module_zip_path.exists()

    def test_preserve_modules_no_copy(self):
        module_zip_path = Path(self.tmpdir.name, "module.zip")
        with ZipFile(module_zip_path, "w") as zf:
            zf.writestr("module.json", "Hello world")

        with TemporaryDirectory() as td:
            td_path = Path(td, "test2")
            extract_zip.extract_zip_files(
                Path(self.tmpdir.name), 
                out_dir=td_path, 
                should_execute=True,
                copy_modules=False
            )
            outpath = Path(td_path, "test", "test.txt")
            assert outpath.exists()
            assert self.test_zip.exists()

            zip_path = Path(td_path, "modules", "module.zip")
            assert zip_path.exists()
            assert not module_zip_path.exists()
