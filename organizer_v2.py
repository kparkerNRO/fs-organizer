from collections import defaultdict
from virtual_folder import VirtualFile, build_folder_structure, VirtualFolder
from pathlib import Path
import click
import os
import re
from pprint import pprint

from common import VIEW_TYPES

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
    """
    Trims out standard view type suffixes from a filename
    """
    f_suffix = None
    f_name = base_name
    for suffix in VIEW_TYPES:
        if base_name.endswith(suffix):
            f_suffix = suffix.strip()
            f_name = _strip_part_from_base(base_name, suffix)

    return f_suffix, f_name


def _parse_name(name, final_token, has_suffix, use_suffix=False):
    """
    Generate the new file name after cleaning off extra tokens.
    If there is no new file name, return "base"
    """
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


def group_similar_folders(folder: VirtualFolder):
    """
    calculates if any subfolders in the supplied folder can be grouped
    into similar folders (eg: snow day and snow night become snow/day and snow/night)

    returns a mapping of file name to new folder stucture
    """
    # map basename to file, and only work on the base names
    pathnames_to_path = {
        subpath.source_path.name: subpath for subpath in folder.subfolders.values()
    }
    base_paths = list(pathnames_to_path.keys())
    # print(f"Grouping {path}")

    names_to_replace = {subpath: [subpath] for subpath in base_paths}
    token_map = dict()

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

    # if len(token_map) > 0:
    #     print()
    #     print(folder.name)
    #     print("Will group the following folders:")
    #     pprint(token_map)

    # print("actual renames")
    # pprint(names_to_replace)

    return names_to_replace


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


def list_to_dict(dict_out, lst, file):
    """
    convert a list into a heirarchical dictionary
    """
    # print(dict_out)
    if not lst:
        return file
    else:
        head, *tail = lst
        # print(head)
        # print(tail)
        dict_out[head] = list_to_dict({}, tail, file)

        # pprint(dict_out)
        # print()
        return dict_out


def dict_to_virtualfs_nodes(input_dict: dict):
    """
    given a dictionary that represents a new virtual FS structure,
    create a list of virtual FS nodes mapping to the keys of the input dict,

    if a supplied folder has only a single file in it, the file is moved up
    a level and the folder is dropped
    """

    # pprint(input_dict)
    output_list = []
    for name, subtree in input_dict.items():
        # print(f"Key: {key}")
        # print(f"Values: {values}")
        try:
            root_folder = VirtualFolder(None, name)

            for subtree in subtree:
                subfolders = dict_to_virtualfs_nodes(subtree)

                for folder in subfolders:
                    root_folder.add_virtual_subfolder(folder)

            output_list.append(root_folder)
        except TypeError:
            # print("hit exception")
            # print(f"values: {values}")

            # presumably it's the bottom file
            if isinstance(subtree, VirtualFolder):
                root_folder = VirtualFolder(subtree.source_path, name)

                # if there is only one subfolder, drop the subfolder, and
                # move its contents up
                if len(subtree.subfolders) == 1:
                    next_value = list(subtree.subfolders.values())[0]
                    if isinstance(next_value, VirtualFile):
                        root_folder.add_virtual_subfolder(next_value)
                    else:
                        for _, folder in next_value.subfolders.items():
                            root_folder.add_virtual_subfolder(folder)
                elif len(subtree.subfolders) == 0:
                    # if there are no subfolders, drop the entry entirely
                    continue
                else:
                    for _, folder in subtree.subfolders.items():
                        root_folder.add_virtual_subfolder(folder)
            else:
                root_folder = VirtualFile(subtree.source_path)

            # print(f"new base folder: {root_folder}")
            output_list.append(root_folder)

        # print()

    # print(f"Returning: {output_dict}")
    # pprint(output_dict)
    return output_list


def reorganize_virtualfs(virtual_fs: VirtualFolder):
    """
    creates a new virtual fs based on the existing one, where
    folder names that include a path seperator get exploded into
    subfiles
    """

    # pprint(virtual_fs.get_folders_dict())

    if isinstance(virtual_fs, VirtualFile):
        return virtual_fs

    # create the new hierarchy
    new_structure = defaultdict(list)
    for subfile in virtual_fs.subfolders.values():
        if os.sep in subfile.name:
            parts = subfile.name.split(os.sep)
            head, *tail = parts
            data = list_to_dict({}, tail, subfile)
            new_structure[head].append(data)
        elif not subfile.name:
            if isinstance(subfile, VirtualFolder):
                for grandfile in subfile.subfolders.values():
                    new_grandfile = reorganize_virtualfs(grandfile)
                    new_structure[new_grandfile.name] = new_grandfile
            else:
                new_structure["Unknown"] = grandfile
        else:
            # print(f"\t{subfile.name}")
            new_structure[subfile.name] = subfile
    # pprint(new_structure)
    # print()

    # create the new folder structure
    # print("building new structure")
    new_fs_list = dict_to_virtualfs_nodes(new_structure)

    new_root = VirtualFolder(virtual_fs.source_path, virtual_fs.name)
    for new_subfolder in new_fs_list:
        if isinstance(new_subfolder, VirtualFolder):
            new_subfolder = reorganize_virtualfs(new_subfolder)
        new_root.add_virtual_subfolder(new_subfolder)

    # pprint(new_root.get_folders_dict())

    return new_root


def clean_filename(base_name, full_path):
    """
    Handles a bunch of standardized cleanup for junk that ends up in folder names
    """
    # print(f"\tstarting with '{full_path}' - '{base_name}'")
    out_dir_name = base_name
    source_dir = Path(full_path)
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
    # if not parent_len:
    #     index = _get_creator_index(parts)
    # else:
    #     index = parent_len
    # creator_parts = list(parts)[index:]

    for part in parts:
        if part in parts:
            # print(f"Stripping {part} from {out_dir_name}")
            out_dir_name = _strip_part_from_base(out_dir_name, part)
            # print(f"got {out_dir_name}")

    # remove any creator-specific removals
    for _, removes in creator_removes.items():
        # if creator in str(source_dir) and removes != "":
        if removes != "":
            out_dir_name = _strip_part_from_base(out_dir_name, removes)

    # remove "part" naming
    out_dir_name = re.sub("\s*Pt(\.)?\s*\d\s*", "", out_dir_name)
    out_dir_name = re.sub("\s*Part\s*\d*", "", out_dir_name)
    out_dir_name = re.sub("\/\s*\d+\s*", "", out_dir_name)
    out_dir_name = out_dir_name.strip(" ()")

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

    # print(f"\treturning '{out_dir_name}'")

    return out_dir_name


def clean_folder_names(virtual_fs, working_path):
    """
    Standardize file names and remove duplicate terms from the path
    """

    for subfile in virtual_fs.subfolders.values():
        if subfile.is_dir():
            # print(f"cleaning {subfile.name} in '{working_path}'")
            sub_name = subfile.name
            cleaned_name = clean_filename(sub_name, working_path)
            # print(f"\t{cleaned_name}")
            subfile.name = cleaned_name
            clean_folder_names(subfile, os.path.join(working_path, sub_name))


def count_terms(virtual_fs, terms_counter):
    for subfile in virtual_fs.subfolders.values():
        if subfile.is_dir():
            terms_counter[subfile.name] += 1
            count_terms(subfile, terms_counter)


def flatten_base_entries(virtual_fs):
    """
    "base" folders should be at the bottom of the stack. If they
    containe subfolders, those folders should be promoted
    """
    folders_to_promote = []

    for subfile in virtual_fs.subfolders.values():
        keys_to_drop = set()
        if subfile.is_dir():
            if subfile.name == "Base" and isinstance(subfile, VirtualFolder):
                for key, grandfile in subfile.subfolders.items():
                    if grandfile.is_dir():
                        folders_to_promote.append(grandfile)
                        keys_to_drop.add(key)
            flatten_base_entries(subfile)
            
            for key in keys_to_drop:
                subfile.subfolders.pop(key)
    
    for folder in folders_to_promote:
        virtual_fs.add_virtual_subfolder(folder)



def reorganize_tree(virtual_fs: VirtualFolder, terms_counter):

    root = virtual_fs

    new_children = defaultdict(list)
    children_to_remove = set()

    # promote any grandchildren who outrank the child
    for index_name, subfile in root.subfolders.items():
        if subfile.is_dir():
            file_score = terms_counter[subfile.name]
            grandchildren_to_remove = set()
            for grand_index_name, grandsubfile in subfile.subfolders.items():
                if grandsubfile.is_dir():
                    grandname = grandsubfile.name
                    grandscore = terms_counter[grandname]
                    if grandscore < file_score:
                        # promote the grandchild to a child, and insert a new generation
                        new_children[grandname].append(grandsubfile)
                        grandsubfile.insert_intermediate_folder(subfile.name)
                        grandchildren_to_remove.add(grand_index_name)
            for granchild in grandchildren_to_remove:
                subfile.subfolders.pop(granchild)
            if len(subfile.subfolders) == 0:
                # child is now an empty folder, and we remove it
                children_to_remove.add(index_name)

    for child in children_to_remove:
        root.subfolders.pop(child)

    # for our new children, merge any duplicate folder names
    flattend_children = []
    for _, new_child in new_children.items():
        if len(new_child) > 1:
            # subfiles = [subfile for child in new_child for subfile in child.subfiles.values()]
            # flat_list = [item for sublist in list_of_lists for item in sublist]

            subfiles = [child.subfolders.values() for child in new_child]
            subfiles = [item for sublist in subfiles for item in sublist]
            replaced_child = VirtualFolder(new_child[0].source_path, new_child[0].name)
            for subfile in subfiles:
                replaced_child.add_virtual_subfolder(subfile)
            flattend_children.append(replaced_child)
        else:
            flattend_children.append(new_child[0])

    # add the new children to our list
    for new_child in flattend_children:
        root.add_virtual_subfolder(new_child)

    # recurse through the children
    children_to_remove = set()
    children_to_add = set()
    for subfile in root.subfolders.values():
        if subfile.is_dir():
            reorganize_tree(subfile, terms_counter)

            # after re-organizing, if there's only one folder at the next
            # level, remove it
            if len(subfile.subfolders) == 1:
                children_to_add.add(list(subfile.subfolders.values())[0])
                children_to_remove.add(subfile.name)

    for remove in children_to_remove:
        root.subfolders.pop(remove)
    for add in children_to_add:
        root.add_virtual_subfolder(add)


def organize_fs(source: Path):
    import json

    print("Step 0: build the virtual FS")
    virtual_fs = build_folder_structure(source)

    # step 1: group folders + files by similar terms (eg day, night, clean, etc)
    print("Step 1: group files")
    organize_groups(virtual_fs)
    virtual_fs = reorganize_virtualfs(virtual_fs)
    print(f"Total files: {virtual_fs.count_files()}")
    print(json.dumps(virtual_fs.get_folders_dict(), indent=4))

    # step 2: remove redundant terms and folders
    print("\n\n------------")
    print("Step 2: clean files")
    clean_folder_names(virtual_fs, virtual_fs.name)
    virtual_fs = reorganize_virtualfs(virtual_fs)
    # flatten_base_entries(virtual_fs)
    print(f"Total files: {virtual_fs.count_files()}")
    print(json.dumps(virtual_fs.get_folders_dict(), indent=4))

    # step 3: organize the folders by term likelyhood
    print("\n\n------------")
    print("Step 3: reorganize")
    term_counter = defaultdict(int)
    count_terms(virtual_fs, term_counter)
    reorganize_tree(virtual_fs, term_counter)
    print(f"Total files: {virtual_fs.count_files()}")
    print(json.dumps(virtual_fs.get_folders_dict(), indent=4))

    # step 4: profit?


@click.command()
@click.argument("path")
# @click.option("--creators", is_flag=True, default=False)
@click.option("--exec", is_flag=True, default=False)
@click.option("--print_out", is_flag=True, default=False)
def organize(path, exec, print_out):

    path_obj = Path(path)
    organize_fs(path_obj)

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
