from enum import Enum
from pathlib import Path
import shutil
import filecmp
import os
import logging

logger = logging.getLogger(__name__)


ZipBackupState = Enum("ExecBackupState", ["KEEP", "MOVE", "DELETE"])
FileBackupState = Enum("FileBackupState", ["COPY", "MOVE", "IN_PLACE"])

VIEW_TYPES = {"Print", "VTT", "Key & Design Notes"}


class FileMoveException(Exception):
    pass


def _get_files_in_dir(dir):
    if isinstance(dir, str):
        dir = Path(dir)
    return [p for p in dir.iterdir()]


def delete_empty_dir(dir: Path, should_execute):
    next_sub_dir = _get_files_in_dir(dir)
    if len(next_sub_dir) == 0 and ("_" not in str(dir)):
        logger.info(f"would delete {dir}")
        if should_execute:
            dir.rmdir()


def print_path_operation(operator, from_path, to_path=None, should_execute=False):
    if not should_execute:
        logger.info(f"would {operator}:\n\t'{from_path}'")
        if to_path:
            logger.info(f"\t\tto \n\t'{to_path}'")
    else:
        logger.info(f"{operator}ing:\n\t{from_path}")
        if to_path:
            logger.info(f"\t\tto \n\t'{to_path}'")


def try_move_file(source_file: Path, target_dir: Path, should_execute, copy_file=False):
    # print_path_operation("move", source_file, target_dir)
    if should_execute:
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / source_file.name
        if file_path.exists():
            logger.info(f"Found a duplicate file moving \n{source_file}\n\tto\n{target_dir}")
            if not filecmp.cmp(file_path, source_file):
                # TODO support 2+
                basename, ext = os.path.split(file_path)
                new_file_path = basename + "-1." + ext
                logger.info(f"\tFiles are different. Renaming {file_path} to {new_file_path}")
                file_path = target_dir/new_file_path
            else:
                logger.info("\tfiles are identical")
                if not copy_file: 
                    logger.info("\tRemoving original file")
                    source_file.unlink()
                    return
        
        if copy_file:
            shutil.copy2(source_file, file_path)
        else:
            source_file.rename(file_path)

        return file_path


def merge_directories(source_path: Path, dest_path: Path):
    if source_path == dest_path:
        return

    logger.info(f"moving files from \n\t{source_path} \nto \n\t{dest_path}")
    for subfile in source_path.iterdir():
        subfile_name = subfile.name
        if subfile_name == source_path.name:
            merge_directories(subfile, dest_path)
        else:
            new_home = dest_path / subfile.name
            if new_home.exists() and new_home.is_dir():
                merge_directories(source_path / subfile.name, new_home)
            else:
                subfile.rename(new_home)

    source_path.rmdir()
