import click
import os
import re
import zipfile
import pprint

from pathlib import Path

from common import delete_empty_dir, print_path_operation, try_move_file, VIEW_TYPES

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


def _strip_part_from_base(base_name, part):
    """
    Removes a string "part" from the file name, removing any spaces or additional
    folders (eg, avoids returning <path>//<name>)
    """
    output = base_name.replace(part, "")
    output = re.sub("\s+", " ", output)
    output = re.sub("\/\/", "/", output)
    output = output.strip()

    return output


def _get_creator_index(parts):
    for name in creator_removes.keys():
        try:
            name_index = parts.index(name)
            return name_index
        except ValueError:
            continue

    return 0


def clean_base_name(base_name, source_dir, parent_len=0):
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
            out_dir_name = _strip_part_from_base(out_dir_name, part)
            # print(f"got {out_dir_name}")

    # remove any creator-specific removals
    for creator, removes in creator_removes.items():
        if creator in str(source_dir) and removes != "":
            out_dir_name = _strip_part_from_base(out_dir_name, removes)

    # remove "part" naming
    out_dir_name = re.sub("\s*Pt(\.)?\s*\d\s*", "", out_dir_name)
    out_dir_name = re.sub("\s*Part\s*\d*", "", out_dir_name)
    out_dir_name = re.sub("\/\s*\d+\s*", "", out_dir_name)
    out_dir_name = out_dir_name.strip(" ()")

    # remove file dimensions
    # out_dir_name = re.sub("(\[)?\d+x\d+(\])?", "", out_dir_name)

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
        final_path.mkdir()

    for member in zipref.namelist():
        if member.startswith("__MACOSX") or ".DS_Store" in member:
            continue
        # print(f"Extracting {member}")
        try:
            zipref.extract(member, final_path)
        except zipfile.BadZipFile:
            print(f"Bad Zip file for file {member}")
            continue


def merge_directories(source_path: Path, dest_path: Path):
    if source_path == dest_path:
        return

    print(f"moving files from \n\t{source_path} \nto \n\t{dest_path}")
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


def extract_zip(zip_file: Path, should_execute=True):
    """
    Extracts a zip file to a similarly named directory and deletes the zip file
    """
    print(f"\npreparing to unzip {zip_file}")

    dir = zip_file.parent
    current_path = Path(dir)
    filename = os.path.splitext(zip_file.name)[0]

    # preprocess the name
    out_dir_name = clean_base_name(filename, dir)
    print(f"target dir: {out_dir_name}")

    # do the work
    final_path = Path(dir, out_dir_name)

    with zipfile.ZipFile(zip_file, "r") as zipref:
        zipPath = zipfile.Path(zipref)
        zip_children = [p.name for p in zipPath.iterdir()]
        if filename in zip_children:
            # print(f"Zipfile {zip_file} has a redundant directory")
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

        # if should_execute:
        # zip_file.unlink()
    print()
    return final_path


def extract_zip_files(path: Path, should_execute: bool):
    # print(f"Looking for zips in {path}")
    for p in path.iterdir():
        if p.is_dir():

            extract_zip_files(p, should_execute)
        else:
            if p.suffix == ".zip":
                zip_path = extract_zip(p, should_execute)
                if should_execute:
                    extract_zip_files(zip_path, should_execute)


def print_dir(path: Path):
    for subfile in path.iterdir():
        if subfile.is_dir():
            if subfile.name == ".ts":
                continue
            print_dir(subfile)
            print()
        else:
            print(subfile)


@click.command()
@click.argument("path")
@click.option("--zip", is_flag=True, default=False)
@click.option("--exec", is_flag=True, default=False)
def list_path(path, zip, exec):
    # Pass 1 - unzip files
    # pass 2 - cleanup path names
    # pass 3 - re-organize folders, move final categorization to the bottom layer
    path_obj = Path(path)

    if zip:
        extract_zip_files(path_obj, exec)


if __name__ == "__main__":
    list_path()
