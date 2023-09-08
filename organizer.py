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


def _get_max_token_overlap(tokens, name_to_comp):
    base_token = tokens[0]
    working_token = base_token

    # greedily add tokens until they stop matching
    for i in range(1, len(tokens) + 1):
        test_token = " ".join(tokens[0:i])
        if name_to_comp.startswith(test_token):
            working_token = test_token
        else:
            break

    return working_token


def _split_suffix(base_name):
    f_suffix = None
    f_name = base_name
    for suffix in VIEW_TYPES:
        if base_name.endswith(suffix):
            f_suffix = suffix.strip()
            f_name = _strip_part_from_base(base_name, suffix)

    return f_suffix, f_name


def _parse_name(name, final_token, has_suffix, use_suffix=False):
    next_name = _strip_part_from_base(name, final_token)
    next_name = next_name.strip(PATH_EXTRAS)
    # print(f"next name: '{next_name}'")
    if next_name == "":

        # if we _dont_ want to take the suffix into account
        # always
        if not (has_suffix and use_suffix):
            next_name = "Base"

        # print(f"Given: {name} /token: {final_token} -> '{next_name}'")
    # print(f"Has suffix: {has_suffix}. Uses it? {use_suffix}")

    # print(f"next name: \t'{next_name}'")

    return next_name


def group_similar_folders(path):
    # map basename to file, and only work on the base names
    pathnames_to_path = {subpath.name: subpath for subpath in path.iterdir()}
    base_paths = list(pathnames_to_path.keys())
    # print(f"Grouping {path}")

    # names_to_replace = {}
    names_to_replace = {subpath: [subpath] for subpath in base_paths}
    token_map = {}

    for base_name, file in pathnames_to_path.items():
        if not file.is_dir():
            continue

        if not base_name in base_paths:
            continue

        base_paths.remove(base_name)
        # print(f"processing {base_name}")

        working_path = []

        # pull out the type suffix
        working_suffix, new_name = _split_suffix(base_name)
        if working_suffix:
            working_path.append(working_suffix)

        tokens = new_name.split(" ")
        first_token = tokens[0]
        final_token = None

        # iterate through the names, and find any matches
        for working_name in base_paths:
            if not pathnames_to_path[working_name].is_dir():
                continue

            tmp_suffix, next_name = _split_suffix(working_name)
            has_suffix = tmp_suffix is not None
            if tmp_suffix is None:
                tmp_suffix = ""

            if final_token and next_name.startswith(final_token):
                token_map[final_token].add(working_name)
                next_name = _parse_name(next_name, final_token, has_suffix)
                names_to_replace[working_name] = [tmp_suffix, final_token, next_name]

            elif next_name.startswith(first_token):
                computed_token = _get_max_token_overlap(tokens, next_name)
                if computed_token != first_token and computed_token not in (
                    "City of",
                    "Lair of the",
                ):
                    final_token = computed_token
                    # print(f"selecting token {computed_token}")
                    token_map[final_token] = {base_name, working_name}

                    next_name = _parse_name(next_name, final_token, has_suffix)
                    names_to_replace[working_name] = [
                        tmp_suffix,
                        final_token,
                        next_name,
                    ]

                    new_name = _parse_name(
                        new_name, final_token, working_suffix is not None
                    )
                    names_to_replace[base_name] = working_path + [final_token, new_name]

        # at the end of an iteration, remove all the matches from the working list
        if final_token:
            for entry in token_map[final_token]:
                if entry in base_paths:
                    base_paths.remove(entry)

    # at the end of the set, see if we accidentally grabbed two overlapping sets
    for key in token_map.keys():
        for key2 in token_map.keys():
            if key in key2 and not key == key2:
                for base_name in token_map[key2]:
                    names_to_replace[2] = key
                token_map[key] |= token_map[key2]
                token_map[key2] = {}

    # if everyone is a base, no one is
    for key, files in token_map.items():
        files_in_set = [names_to_replace[file] for file in files]
        all_base = [v[2] == "Base" if len(v) == 3 else False for v in files_in_set]
        res = all(all_base)

        # print(f"Sets to process {files_in_set}")
        # print(f"All base {all_base}")
        # print(f"Results {res}")

        if res:
            for item in files_in_set:
                item[2] = ""

    if len(token_map) > 0:
        print()
        print(path)
        print("Will group the following folders:")
        pprint.pprint(token_map)

    # print("actual renames")
    # pprint.pprint(names_to_replace)

    return names_to_replace


def organize_folders(
    true_path: Path, working_path: Path, should_execute: bool, start_path_len=0
) -> list[Path]:
    # print(f"Path: {path}")
    names_to_replace = group_similar_folders(true_path)
    # pprint.pprint(names_to_replace)

    printed_rename = False

    duplicate_files = []
    for subfile in true_path.iterdir():
        if subfile.is_dir():

            print(f"subfile: {subfile}")
            # print(f"working path: {final_path}")

            sub_name = subfile.name
            print(sub_name)

            sub_paths = names_to_replace[sub_name]
            # print(f"Cleaning {sub_paths}")
            pprint.pprint(sub_paths)
            proc_paths = [
                clean_base_name(sub_path, working_path, start_path_len)
                for sub_path in sub_paths
            ]
            cleaned_name = os.path.join(*proc_paths)

            print(cleaned_name)
            print()


            next_name = working_path / cleaned_name
            duplicate_files.extend(organize_folders(subfile, next_name, should_execute))

            # see if the subdir is empty, and if so, delete it
            delete_empty_dir(subfile, should_execute)
        else:
            if true_path != working_path:
                if not printed_rename:
                    print_path_operation(
                        "rename", true_path, working_path, should_execute
                    )
                    printed_rename = True

                if result := try_move_file(subfile, working_path, should_execute):
                    duplicate_files.append(result)

        # if printed_rename:
        #     cont = input("Keep going? y/n")
        #     if cont.lower() == "n":
        #         exit()

    return duplicate_files


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
# @click.option("--creators", is_flag=True, default=False)
@click.option("--exec", is_flag=True, default=False)
@click.option("--print_out", is_flag=True, default=False)
def list_path(path, zip, exec, print_out):
    # Pass 1 - unzip files
    # pass 2 - cleanup path names
    # pass 3 - re-organize folders, move final categorization to the bottom layer
    path_obj = Path(path)
    duplicates = []

    if zip:
        extract_zip_files(path_obj, exec)
    else:
        start_len = len(path_obj.parts) - 2
        duplicates = organize_folders(path_obj, Path(path), exec, start_len)

    if print_out:
        print_dir(path_obj)

    if duplicates:
        print()
        print("------ Duplicates -------")
        for file in duplicates:
            print(f"tried to move:\n\t'{file[0]}'")
            print(f"\t\tto \n\t'{file[1]}'")


if __name__ == "__main__":
    list_path()
