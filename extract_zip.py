import click
import os
import re
import zipfile
import pprint

from pathlib import Path

from common import (
    ZipBackupState,
    FileMoveException,
    merge_directories,
    print_path_operation,
    try_move_file,
)
from filename_utils import clean_filename, clean_path

PATH_EXTRAS = " -,()/"

creator_removes = {
    "CzePeku": "$5 Rewards",
    "Limithron": "Admiral",
    "The Reclusive Cartographer": "_MC",
    "Baileywiki": "",
    "Unknown": "",
    # "Samples": "",
    "Caeora": "",
    "DWW": "",
    "MAD Cartographer": "",
    "MikWewa": "$5 Map Rewards",
    "Tom Cartos": "Tier 2+",
}

exceptions = {"The Clean": "Clean", "The": ""}
replace_exceptions = {
    "ItW": "",
}


def extract_zip_without_hidden_files(zipref, final_path):
    if not final_path.exists():
        final_path.mkdir(parents=True)

    for member in zipref.namelist():
        if member.startswith("__MACOSX") or ".DS_Store" in member:
            continue
        try:
            zipref.extract(member, final_path)
        except zipfile.BadZipFile:
            print(f"Bad Zip file for file {member}")
            continue


def extract_zip(
    zip_file: Path,
    out_dir: str = "",
    should_execute=True,
    module_dir="",
    preserve_modules=True,
    copy_modules=True,
):
    """
    Extracts a zip file to a similarly named directory and deletes the zip file
    """
    print(f"\npreparing to unzip {zip_file} to {out_dir}")

    dir = zip_file.parent
    current_path = Path(dir)
    filename = os.path.splitext(zip_file.name)[0]

    # preprocess the name
    out_dir_name = clean_filename(
        filename, creator_removes, exceptions, replace_exceptions
    )
    out_dir_name = clean_path(out_dir_name, str(dir))
    print(f"target dir: {out_dir_name}")

    # do the work
    final_path = Path(out_dir, out_dir_name)

    with zipfile.ZipFile(zip_file, "r") as zipref:
        zipPath = zipfile.Path(zipref)
        zip_children = [p.name for p in zipPath.iterdir()]
        if preserve_modules and "module.json" in zip_children:
            print(f"File {filename} is a foundry module, moving to {module_dir}")
            try_move_file(
                zip_file,
                Path(module_dir),
                should_execute=should_execute,
                copy_file=copy_modules,
            )
            return
        elif filename in zip_children:
            if should_execute:
                # extract to current dir then rename to the target path
                extract_zip_without_hidden_files(zipref, current_path)
                path = current_path / filename
                if final_path.exists():
                    merge_directories(path, final_path)
                else:
                    print(f"moving \n\t{path} \nto \n\t{final_path}")
                    path.rename(final_path)
        else:
            print_path_operation("extract", zip_file, should_execute=should_execute)
            if should_execute:
                extract_zip_without_hidden_files(zipref, final_path)
    return final_path


def extract_zip_files(
    path: Path,
    out_dir: str = "",
    should_execute: bool = False,
    zip_backup_state: ZipBackupState = ZipBackupState.KEEP,
    zip_backup_dir: str = "",
    module_dir=None,
    preserve_modules=True,
    copy_modules=True,
):
    if not out_dir:
        out_dir = path
    elif isinstance(out_dir, str) and len(out_dir) == 0:
        out_dir = path
    elif isinstance(out_dir, str):
        out_dir = Path(out_dir)

    if not zip_backup_dir:
        zip_backup_dir = ""
    elif (
        zip_backup_dir and isinstance(zip_backup_dir, str) and len(zip_backup_dir) != 0
    ):
        zip_backup_dir = Path(zip_backup_dir)

    if not module_dir:
        module_dir = Path(out_dir, "modules")
    elif isinstance(module_dir, str):
        module_dir = Path(module_dir)

    for p in path.iterdir():
        if p.is_dir():
            sub_out_dir = out_dir / p.name
            next_zip_backup_dir = zip_backup_dir / p.name
            next_module_dir = module_dir / p.name
            extract_zip_files(
                p,
                sub_out_dir,
                should_execute,
                zip_backup_state,
                next_zip_backup_dir,
                module_dir=next_module_dir,
                preserve_modules=preserve_modules,
                copy_modules=copy_modules,
            )
        elif p.suffix == ".zip":
            # if the file contains zipfiles, unzip those too
            zip_path = extract_zip(
                p,
                out_dir,
                should_execute,
                module_dir=module_dir,
                preserve_modules=preserve_modules,
                copy_modules=copy_modules,
            )
            if should_execute and zip_path:
                out_dir.mkdir(exist_ok=True)
                extract_zip_files(
                    zip_path,
                    out_dir,
                    should_execute,
                    zip_backup_state,
                    zip_backup_dir,
                    module_dir=module_dir,
                    preserve_modules=preserve_modules,
                    copy_modules=copy_modules,
                )

                if zip_backup_state == ZipBackupState.KEEP:
                    # do nothing
                    pass
                elif zip_backup_state == ZipBackupState.MOVE:
                    # move the zip to the new directory, and remove the current one
                    if not zip_backup_dir:
                        raise FileMoveException(
                            f"Cannot move zipfiles with no specified output directory"
                        )
                    zip_backup_dir.mkdir(exist_ok=True, parents=True)
                    try_move_file(p, zip_backup_dir, should_execute)
                elif zip_backup_state == ZipBackupState.DELETE:
                    # remove the current one
                    p.unlink()
                else:
                    raise FileMoveException(
                        f"Unknown zip backup state {zip_backup_state}"
                    )


# def _print_dir(path: Path):
#     for subfile in path.iterdir():
#         if subfile.is_dir():
#             if subfile.name == ".ts":
#                 continue
#             _print_dir(subfile)
#             print()
#         else:
#             print(subfile)


@click.command()
@click.argument("path")
@click.option("--exec", "-e", is_flag=True, default=False)
@click.option("--output", default="")
@click.option("--backup_dir", default="")
@click.option("--delete_files", "-D", is_flag=True, default=False)
def process_input(path, exec, output, backup_dir, delete_files):
    # Pass 1 - unzip files
    # pass 2 - cleanup path names
    # pass 3 - re-organize folders, move final categorization to the bottom layer
    path_obj = Path(path)

    exec_format = ZipBackupState.KEEP
    if delete_files:
        if backup_dir:
            exec_format = ZipBackupState.MOVE
        else:
            exec_format = ZipBackupState.DELETE

    extract_zip_files(
        path_obj,
        should_execute=exec,
        out_dir=output,
        zip_backup_dir=backup_dir,
        zip_backup_state=exec_format,
    )


if __name__ == "__main__":
    process_input()
