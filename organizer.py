from collections import defaultdict
from virtual_folder import (
    InsertException,
    VirtualFile,
    build_folder_structure,
    VirtualFolder,
)
from pathlib import Path
import click
import os

from common import ExecBackupState, FileMoveException, try_move_file

from filename_utils import (
    split_view_type,
    process_file_name,
    get_max_common_string,
    clean_filename,
    clean_path,
)


creator_removes = {
    "CzePeku": "$5 Rewards",
    "Limithron": "Admiral",
    "The Reclusive Cartographer": "_MC",
    "Baileywiki": "",
    "Unknown": "",
    "Caeora": "",
    "DWW": "",
    "MAD Cartographer": "",
    "MikWewa": "$5 Map Rewards",
    "Tom Cartos": "Tier 2+",
}

file_name_exceptions = {"The Clean": "Clean", "The": ""}
replace_exceptions = {
    "ItW": "",
}
grouping_exceptions = (
    "City of",
    "Lair of the",
    "Tower of",
    "The ruins of",
    "The",
    "Bone",
    "Wizard",
    "War",
)


class FSException(Exception):
    def __init__(self, invalid_file) -> None:
        self.invalid_file = invalid_file


def generate_path(view_type, token, working_name):
    """
    joins the components of a grouping path, stripping out any cases
    where one of the supplied variables is '' or None
    """
    base_path = [view_type, token, working_name]
    return [x for x in base_path if x]


def group_similar_names(names_to_group: list):
    names_to_replacement = {subpath: [subpath] for subpath in names_to_group}
    token_to_filenames = dict()

    names_to_process = list(names_to_group)
    for base_name in names_to_group:
        # handle the data structure mismatch where one's already been removed
        if not base_name in names_to_process:
            continue

        names_to_process.remove(base_name)

        # pull out the type suffix
        initial_suffix, parsed_base_name = split_view_type(base_name)

        tokens = parsed_base_name.split(" ")
        first_token = tokens[0]
        working_token = None

        # iterate through the names, and find any matches
        for working_name in names_to_process:
            working_suffix, next_name = split_view_type(working_name)
            has_suffix = working_suffix is not None

            if working_token and next_name.startswith(working_token):
                # main case: store a new token match
                token_to_filenames[working_token].add(working_name)

                root_name = process_file_name(next_name, working_token, has_suffix)
                names_to_replacement[working_name] = generate_path(
                    working_suffix, working_token, root_name
                )

            elif next_name.startswith(first_token):
                # base case, initialize a new token
                computed_token = get_max_common_string(tokens, next_name)

                # if the token is in the exceptions list, don't group
                if computed_token not in grouping_exceptions:
                    # if we've found a token with overlap, save it
                    working_token = computed_token

                    # record that we have a matching token
                    token_to_filenames[working_token] = {base_name, working_name}

                    # store the new path name mapping
                    root_name = process_file_name(next_name, working_token, has_suffix)
                    names_to_replacement[working_name] = generate_path(
                        working_suffix, working_token, root_name
                    )

                    # store the existing path name mapping
                    parsed_base_name = process_file_name(
                        parsed_base_name, working_token, initial_suffix is not None
                    )
                    names_to_replacement[base_name] = generate_path(
                        initial_suffix, working_token, parsed_base_name
                    )
            elif has_suffix:
                pass

        # at the end of an iteration, remove all the matches from the working list
        if working_token:
            for entry in token_to_filenames[working_token]:
                if entry in names_to_process:
                    names_to_process.remove(entry)

    # at the end of the set, see if we accidentally grabbed two overlapping sets
    tokens_to_revisit = {}
    for token1 in token_to_filenames.keys():
        for token2 in token_to_filenames.keys():
            # if this isn't itself, and the main token is a subset of the other token
            if token1 in token2 and not token1 == token2:
                # replace the longer token with the shorter one in the replacement array
                delta = token2.replace(token1, "")
                for filename in token_to_filenames[token2]:
                    working_path = names_to_replacement[filename]
                    working_path[-1] = (delta + " " + working_path[-1]).strip()
                    working_path[-2] = token1

                # empty the longer token and transfer it's contents to the shorter one
                token_to_filenames[token1] |= token_to_filenames[token2]
                token_to_filenames[token2] = set()
                tokens_to_revisit[token1] = token_to_filenames[token1]

    # if everyone is a base, no one is
    for _, files in token_to_filenames.items():
        files_in_set = [names_to_replacement[file] for file in files]
        all_base = [v[-1] == "Base" for v in files_in_set]
        res = all(all_base)

        if res:
            for item in files_in_set:
                if len(item) > 1:
                    item.remove("Base")

    return names_to_replacement, token_to_filenames


def group_nested_folder(file_to_str_to_group: dict):
    lookback = {value: key for key, value in file_to_str_to_group.items()}
    if len(file_to_str_to_group) == 0:
        return {}

    file_to_replacement, tokens_to_filename = group_similar_names(
        file_to_str_to_group.values()
    )

    true_file_to_replacment = {
        key: file_to_replacement[value] for key, value in file_to_str_to_group.items()
    }
    new_file_to_grouping = true_file_to_replacment

    for token, filenames in tokens_to_filename.items():
        if token:
            file_to_next_grouping = {
                file: file_to_replacement[file][-1] for file in filenames
            }
            file_to_next_group = group_nested_folder(file_to_next_grouping)

            for file, group in file_to_next_group.items():
                if len(group) > 1:
                    full_filename = lookback[file]
                    true_file_to_replacment[full_filename] = (
                        true_file_to_replacment[full_filename][:-1] + group
                    )

    return new_file_to_grouping


def group_similar_folders(folder: VirtualFolder):
    """
    calculates if any subfolders in the supplied folder can be grouped
    into similar folders (eg: snow day and snow night become snow/day and snow/night)

    returns a mapping of file name to new folder stucture
    """
    # extract out the directory names to work on
    names_to_process = [
        subpath.name for subpath in folder.subfolders.values() if subpath.is_dir()
    ]
    files_to_subgroup = {file: file for file in names_to_process}

    new_replacement = group_nested_folder(files_to_subgroup)

    return new_replacement


def organize_groups(virtual_fs: VirtualFolder):
    """
    Organize the files in the file system so that folders with
    similar names are grouped together and split into subfolders
    """

    names_to_replace = group_similar_folders(virtual_fs)

    for subfile in virtual_fs.subfolders.values():
        if subfile.is_dir():
            sub_name = subfile.name
            sub_paths = names_to_replace[sub_name]

            cleaned_name = os.path.join(*sub_paths)

            subfile.name = cleaned_name
            organize_groups(subfile)

    return virtual_fs


def list_to_dict(dict_out, lst, file):
    """
    convert a list into a heirarchical dictionary
    """
    if not lst:
        return file
    else:
        head, *tail = lst
        dict_out[head] = list_to_dict({}, tail, file)

        return dict_out


def update_virtual_folder(data: VirtualFolder, name):
    root_folder = VirtualFolder(data.source_path, name)

    for _, folder in data.subfolders.items():
        root_folder.add_virtual_subfolder(folder)

    return root_folder


def dict_to_virtualfs_nodes(input_dict: dict):
    """
    given a dictionary that represents a new virtual FS structure,
    create a list of virtual FS nodes mapping to the keys of the input dict,

    if a supplied folder has only a single file in it, the file is moved up
    a level and the folder is dropped

    input_dict is expected to be a dict mapping
        str -> VirtualFolder
        str -> dict(str -> list)

    returns a list of VirtualFolder nodes
    """

    output_list = []
    for name, data in input_dict.items():
        if isinstance(data, (VirtualFile, VirtualFolder)):
            if isinstance(data, VirtualFolder):
                root_folder = VirtualFolder(data.source_path, name)

                for _, folder in data.subfolders.items():
                    root_folder.add_virtual_subfolder(folder)

            else:
                root_folder = VirtualFile(data.source_path)

            # print(f"new base folder: {root_folder}")
            output_list.append(root_folder)
        else:
            root_folder = VirtualFolder(None, name)
            subfolders = dict_to_virtualfs_nodes(data)
            for folder in subfolders:
                root_folder.add_virtual_subfolder(folder)

            output_list.append(root_folder)

    return output_list


def reorganize_virtualfs_old(virtual_fs: VirtualFolder):
    """
    creates a new virtual fs based on the existing one, where
    folder names that include a path seperator get exploded into
    subfiles
    """

    if isinstance(virtual_fs, VirtualFile):
        return virtual_fs

    # create the new hierarchy
    new_structure = defaultdict(dict)
    for subfile in virtual_fs.subfolders.values():
        # if the name has a directory structure in it, explode the structure
        if os.sep in subfile.name:
            parts = subfile.name.split(os.sep)
            head, *tail = parts
            list_to_dict(new_structure[head], tail, subfile)
        # if the subfile name is an empty string,
        # we should promote the grandchild to this level
        elif not subfile.name:
            if isinstance(subfile, VirtualFolder):
                for grandfile in subfile.subfolders.values():
                    new_grandfile = reorganize_virtualfs_old(grandfile)
                    new_structure[new_grandfile.name] = new_grandfile
            else:
                # or something's gone very wrong
                raise FSException(grandfile)
        else:
            new_structure[subfile.name] = subfile

    # create the new folder structure
    new_fs_list = dict_to_virtualfs_nodes(new_structure)

    new_root = VirtualFolder(virtual_fs.source_path, virtual_fs.name)
    for new_subfolder in new_fs_list:
        if isinstance(new_subfolder, VirtualFolder):
            new_subfolder = reorganize_virtualfs_old(new_subfolder)
        new_root.add_virtual_subfolder(new_subfolder)

    return new_root


def reorganize_virtualfs(virtual_fs: VirtualFolder, new_root: VirtualFolder = None):
    if not new_root:
        new_root = VirtualFolder(virtual_fs.source_path, virtual_fs.name)

    for subname, subfile in virtual_fs.subfolders.items():
        if isinstance(subfile, VirtualFile):
            new_root.add_virtual_subfolder(subfile)
        elif not subfile.name:
            reorganize_virtualfs(subfile, new_root)
        else:
            working_folder = new_root

            if os.sep in subfile.name:
                parts = subfile.name.split(os.sep)
                for part in parts:
                    if part not in working_folder.subfolders:
                        new_folder = VirtualFolder(None, part)
                        working_folder.add_virtual_subfolder(new_folder)
                    working_folder = working_folder.subfolders[part]

            else:
                if subfile.name not in working_folder.subfolders:
                    new_folder = VirtualFolder(None, subfile.name)
                    working_folder.add_virtual_subfolder(new_folder)
                working_folder = working_folder.subfolders[subfile.name]

            reorganize_virtualfs(subfile, working_folder)

    return new_root


def clean_folder_names(virtual_fs: VirtualFolder):
    """
    Standardize file names and remove duplicate terms from the path
    """

    for subfile in virtual_fs.subfolders.values():
        if subfile.is_dir():
            sub_name = subfile.name
            cleaned_name = clean_filename(
                sub_name, creator_removes, file_name_exceptions, replace_exceptions
            )
            subfile.name = cleaned_name
            clean_folder_names(subfile)

    return virtual_fs


def remove_duplicate_folder_terms(virtual_fs, working_path):
    for subfile in virtual_fs.subfolders.values():
        if subfile.is_dir():
            sub_name = subfile.name
            cleaned_name = clean_path(sub_name, working_path)
            subfile.name = cleaned_name
            remove_duplicate_folder_terms(subfile, os.path.join(working_path, sub_name))

    return virtual_fs


def count_terms(virtual_fs, terms_counter):
    """
    recursively count the number of times each term appears in the
    folder structure
    """
    for subfile in virtual_fs.subfolders.values():
        if subfile.is_dir():
            terms_counter[subfile.name] += 1
            count_terms(subfile, terms_counter)


def promote_grandchildren(
    root_folder: VirtualFolder, subfolder_name: str, grandchildren_to_promote: list
):
    """
    promotes the named grandchildren from the supplied subfolder to the root folder
    if the subfolder is empty after promotion, deletes it from the root folder
    """
    subfolder = root_folder.subfolders[subfolder_name]
    granchildren = {
        grandname: subfolder.subfolders[grandname]
        for grandname in grandchildren_to_promote
    }
    grandnames_to_remove = []
    for grandname, grandchild in granchildren.items():
        try:
            root_folder.add_virtual_subfolder(grandchild)
            grandnames_to_remove.append(grandname)
        except InsertException as e:
            print("Unable to merge duplicate files. Conflict occured between")
            print(f"\t{e.source_path}")
            print(f"\t{e.target_path}")

    for grandname in grandnames_to_remove:
        subfolder.subfolders.pop(grandname)

    if len(subfolder.subfolders) == 0:
        root_folder.subfolders.pop(subfolder_name)

    return grandnames_to_remove


def remove_extra_folders(virtual_fs):
    """
    cleans up extra folders, removing folders with 1 or no children, and
    cleaning base entries

    "base" folders should be at the bottom of the stack. If they
    containe subfolders, those folders should be promoted
    """

    folders_moved = True

    while folders_moved:
        folders_moved = False
        folders_to_promote = defaultdict(list)
        empty_subfolders = []

        for sub_name, subfile in virtual_fs.subfolders.items():
            if subfile.is_dir():
                if subfile.name == "Base" and isinstance(subfile, VirtualFolder):
                    for key, grandfile in subfile.subfolders.items():
                        if grandfile.is_dir():
                            folders_to_promote[sub_name].append(key)
                elif len(subfile.subfolders) == 1:
                    grandname, grandchild = list(subfile.subfolders.items())[0]
                    if isinstance(grandchild, VirtualFolder):
                        great_grandchildren = list(grandchild.subfolders.keys())
                        promote_grandchildren(subfile, grandname, great_grandchildren)
                elif len(subfile.subfolders) == 0:
                    empty_subfolders.append(sub_name)

                remove_extra_folders(subfile)

        unpromoted_folders = []
        for subfile, grandchildren in folders_to_promote.items():
            promoted_grandchildren = promote_grandchildren(
                virtual_fs, subfile, grandchildren
            )
            if len(promoted_grandchildren) != len(grandchildren):
                # promotion failed, and we need to not count this directory
                current_unpromoted = [
                    x for x in grandchildren if x not in promoted_grandchildren
                ]
                unpromoted_folders.extend(current_unpromoted)

        for unpromoted in unpromoted_folders:
            folders_to_promote.pop(unpromoted)

        # remove any empty directories
        for subname in empty_subfolders:
            virtual_fs.subfolders.pop(subname)

        folders_moved = len(folders_to_promote) > 0 or len(empty_subfolders) > 0

    return virtual_fs


def reorganize_tree(virtual_fs: VirtualFolder, terms_counter):
    """
    reorganizes the file system so that more frequently used terms
    are closer to the bottom

    the goal of this function is to more-or-less standardize the
    folder structure
    """
    root = virtual_fs

    # this is wildly fucking inefficient but it works

    folders_moved = True
    while folders_moved:
        # promote any grandchildren who outrank the child
        folders_moved = False
        folders_to_promote = defaultdict(list)  # subfile_name -> list of grandchildren

        for index_name, subfile in root.subfolders.items():
            if subfile.is_dir():
                file_score = terms_counter[subfile.name]
                for grand_index_name, grandsubfile in subfile.subfolders.items():
                    if grandsubfile.is_dir():
                        grandname = grandsubfile.name
                        grandscore = terms_counter[grandname]
                        if grandscore < file_score:
                            folders_moved = True
                            # promote the grandchild to a child, and insert a new generation
                            grandsubfile.insert_intermediate_folder(subfile.name)
                            folders_to_promote[index_name].append(grand_index_name)
        for subfile_name, grandchildren in folders_to_promote.items():
            promote_grandchildren(root, subfile_name, grandchildren)

        # recurse through the children
        empty_subfolders = []
        for subname, subfile in root.subfolders.items():
            if subfile.is_dir():
                reorganize_tree(subfile, terms_counter)

        # remove any empty directories
        for subname in empty_subfolders:
            root.subfolders.pop(subname)

    return virtual_fs


def move_file(
    file: VirtualFile, dest_path: Path, should_execute=True, should_copy=False
):
    source_path = file.source_path
    destination = Path(dest_path, file.name)
    try_move_file(source_path, destination, should_execute, should_copy)


def move_folder(
    virtual_fs: VirtualFolder, output_dir="", should_execute=True, should_copy=False
):
    output_path = Path(output_dir, virtual_fs.name)

    for _, subfile in virtual_fs.subfolders.items():
        if subfile.is_file():
            move_file(subfile, output_path, should_execute, should_copy)
        else:
            move_folder(subfile, output_path, should_execute, should_copy)


def move_fs(
    virtual_fs: VirtualFolder,
    output_dir="",
    should_execute=True,
    backup_state=ExecBackupState.MOVE,
):
    if backup_state == ExecBackupState.KEEP:  # copy
        if not output_dir:
            raise FileMoveException(
                f"Cannot duplicate data with no specified output directory"
            )
        move_folder(virtual_fs, output_dir, should_execute, should_copy=True)
    elif backup_state == ExecBackupState.MOVE:  # new folder
        if not output_dir:
            raise FileMoveException(
                f"Cannot move data with no specified output directory"
            )
        move_folder(virtual_fs, output_dir, should_execute, should_copy=False)
    elif backup_state == ExecBackupState.DELETE:  # in place
        if output_dir:
            print(
                f"Folder re-org handed an output directory, but requested in-place"
                + "modification. Will ignore output_dir."
            )
        move_folder(virtual_fs, "", should_execute, should_copy=False)


def organize_fs(
    source: Path, output_dir="", should_execute=False, backup_state=ExecBackupState.MOVE
):
    import json

    print("Step 0: build the virtual FS", flush=True)
    virtual_fs = build_folder_structure(source)
    print(f"Total files: {virtual_fs.count_files()}")

    # step 1: cleanup filenames
    print("\n\n------------")
    print("Step 1: clean files", flush=True)
    virtual_fs = clean_folder_names(virtual_fs)
    virtual_fs = reorganize_virtualfs(virtual_fs)
    print(f"Total files: {virtual_fs.count_files()}")
    # print(json.dumps(virtual_fs.get_folders_dict(False, True), indent=4))

    # step 2: group folders + files by similar terms (eg day, night, clean, etc)
    print("Step 2: group files", flush=True)
    virtual_fs = organize_groups(virtual_fs)
    # print(json.dumps(virtual_fs.get_folders_dict(False, True), indent=4))

    virtual_fs = reorganize_virtualfs(virtual_fs)
    print(f"Total files: {virtual_fs.count_files()}")
    # print(json.dumps(virtual_fs.get_folders_dict(False, True), indent=4))

    # print(json.dumps(virtual_fs.get_folders_dict(), indent=4))
    # step 3: cleanup redundant file names
    print("\n\n------------")
    print("Step 3: clean files again", flush=True)
    virtual_fs = remove_duplicate_folder_terms(virtual_fs, virtual_fs.name)
    # print(json.dumps(virtual_fs.get_folders_dict(False, True), indent=4))
    virtual_fs = clean_folder_names(virtual_fs)
    virtual_fs = reorganize_virtualfs(virtual_fs)
    # print(json.dumps(virtual_fs.get_folders_dict(False, True), indent=4))
    print(f"Total files: {virtual_fs.count_files()}")

    print("\n\n------------")
    print("Step 3.5: remove excess base folders", flush=True)
    virtual_fs = remove_extra_folders(virtual_fs)
    print(f"Total files: {virtual_fs.count_files()}")
    # print(json.dumps(virtual_fs.get_folders_dict(False, True), indent=4))

    # step 3: organize the folders by term likelyhood
    print("\n\n------------")
    print("Step 4: reorganize", flush=True)
    term_counter = defaultdict(int)
    count_terms(virtual_fs, term_counter)
    # print(json.dumps(term_counter, indent=4))

    virtual_fs = reorganize_tree(virtual_fs, term_counter)
    virtual_fs = remove_extra_folders(virtual_fs)
    print(f"Total files: {virtual_fs.count_files()}")
    print(json.dumps(virtual_fs.get_folders_dict(False, True), indent=4))

    # step 4: do the work
    move_fs(virtual_fs, output_dir, should_execute, backup_state)


@click.command()
@click.argument("path")
# @click.option("--zip", is_flag=True, default=False)
@click.option("--exec", is_flag=True, default=False)
@click.option("--print_out", is_flag=True, default=False)
def organize(path, exec, print_out):
    path_obj = Path(path)
    organize_fs(path_obj, output_dir=".")

    # # Pass 1 - unzip files
    # # pass 2 - cleanup path names
    # # pass 3 - re-organize folders, move final categorization to the bottom layer
    # path_obj = Path(path)
    # duplicates = []

    # else:
    #     start_len = len(path_obj.parts) - 2
    #     duplicates = organize_folders(path_obj, Path(path), exec, start_len)

    # if print_out:
    #     print_dir(path_obj)

    # if duplicates:
    #     print()
    #     print("------ Duplicates -------")
    #     for file in duplicates:
    #         print(f"tried to move:\n\t'{file[0]}'")
    #         print(f"\t\tto \n\t'{file[1]}'")


if __name__ == "__main__":
    organize()
