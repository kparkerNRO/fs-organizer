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
from filename_utils import clean_filename, clean_path, strip_part_from_base

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


def _get_creator_index(parts):
    for name in creator_removes.keys():
        try:
            name_index = parts.index(name)
            return name_index
        except ValueError:
            continue

    return 0


def _clean_base_name(base_name, source_dir, parent_len=0):
    # print(f"starting with '{source_dir}' - '{base_name}'")

    if not base_name:
        return ""

    out_dir_name = base_name

    # remove myairbrigde tags
    if "myairbridge" in base_name:
        out_dir_name = base_name.replace("myairbridge-", "")

    # convert underscores into spaces or 's'
    if "_" in base_name and base_name != "_Unsorted":
        if "_s" in base_name:
            out_dir_name = base_name.replace("_s", "s")

        else:
            out_dir_name = base_name.replace("_", " ")

    # process duplicate entries in the path
    parts = source_dir.parts
    if not parent_len:
        index = _get_creator_index(parts)
    else:
        index = parent_len
    creator_parts = list(parts)[index:]
    for part in creator_parts:
        if part in out_dir_name:
            # print(f"Stripping {part} from {out_dir_name}")
            out_dir_name = strip_part_from_base(out_dir_name, part)
            # print(f"got {out_dir_name}")

    # remove any creator-specific removals
    for creator, removes in creator_removes.items():
        if creator in str(source_dir) and removes != "":
            out_dir_name = strip_part_from_base(out_dir_name, removes)

    # remove "part" naming
    out_dir_name = re.sub("\s*Pt(\.)?\s*\d\s*", "", out_dir_name)
    out_dir_name = re.sub("\s*Part\s*\d*", "", out_dir_name)
    out_dir_name = re.sub("\/\s*\d+\s*", "", out_dir_name)
    out_dir_name = out_dir_name.strip(" ()")

    # remove file dimensions
    out_dir_name = re.sub("(\[)?\d+x\d+(\])?", "", out_dir_name)

    # remove special case exceptions
    for exception, replace in exceptions.items():
        if out_dir_name == exception:
            out_dir_name = replace
            break

    for exception, replace in replace_exceptions.items():
        if exception in out_dir_name:
            out_dir_name = out_dir_name.replace(exception, replace)

    # remove numbers at the start and end of the name
    out_dir_name = re.sub("^#?\d{0,2}\s*", "", out_dir_name)
    out_dir_name = re.sub("#?\d{0,2}\s*$", "", out_dir_name)

    # out_dir_name = re.sub("#?\d{2}\s*", "", out_dir_name)

    # cleanup whitespace
    out_dir_name = out_dir_name.replace("(", " ")
    out_dir_name = re.sub("\s+", " ", out_dir_name)
    out_dir_name = re.sub("--", " ", out_dir_name)
    out_dir_name = re.sub("-\s+-", " ", out_dir_name)
    out_dir_name = out_dir_name.strip(PATH_EXTRAS)

    # cleanup number only entries
    out_dir_name = re.sub("^\s*\d+\s*$", "", out_dir_name)

    if len(out_dir_name) == 1:
        out_dir_name = ""

    # print(f"ending with '{source_dir}' - '{out_dir_name}'\n")

    return out_dir_name


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


def extract_zip(zip_file: Path, out_dir: str = "", should_execute=True):
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
        if filename in zip_children:
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

    for p in path.iterdir():
        if p.is_dir():
            sub_out_dir = out_dir / p.name
            next_zip_backup_dir = zip_backup_dir / p.name
            extract_zip_files(
                p, sub_out_dir, should_execute, zip_backup_state, next_zip_backup_dir
            )
        elif p.suffix == ".zip":
            # if the file contains zipfiles, unzip those too
            zip_path = extract_zip(p, out_dir, should_execute)
            if should_execute:
                out_dir.mkdir(exist_ok=True)
                extract_zip_files(
                    zip_path, out_dir, should_execute, zip_backup_state, zip_backup_dir
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
