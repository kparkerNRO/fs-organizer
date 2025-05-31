import re
import logging

# from common import VIEW_TYPES

PATH_EXTRAS = " -,()/"
from pathlib import Path

logger = logging.getLogger(__name__)


def strip_part_from_base(base_name: str, part):
    """
    Removes a string "part" from the file name, removing any spaces or additional
    folders (eg, avoids returning <path>//<name>)
    """
    output = base_name.replace(part, "")
    output = re.sub("\s+", " ", output)
    output = re.sub("\/\/", "/", output)
    output = output.strip()

    return output


def get_max_common_string(tokens, name_to_comp):
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


def split_view_type(base_name):
    """
    Trims out standard view type suffixes from a filename
    """
    f_suffix = None
    f_name = base_name
    for suffix in VIEW_TYPES:
        if base_name.endswith(suffix):
            f_suffix = suffix.strip()
            f_name = strip_part_from_base(base_name, suffix)

    return f_suffix, f_name


def process_file_name(name, final_token, has_suffix, use_suffix=False):
    """
    Generate the new file name after cleaning off extra tokens.
    If there is no new file name, return "base"
    """
    next_name = strip_part_from_base(name, final_token)
    next_name = next_name.strip(PATH_EXTRAS)
    if next_name == "":
        # if we _dont_ want to take the suffix into account
        if not (has_suffix and use_suffix):
            next_name = "Base"

    return next_name


def clean_filename(
    base_name, creator_removes, file_name_exceptions, replace_exceptions
):
    """
    Handles a bunch of standardized cleanup for junk that ends up in folder names
    """
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

    # remove any creator-specific removals
    for _, removes in creator_removes.items():
        # if creator in str(source_dir) and removes != "":
        if isinstance(removes, list):
            for remove in removes:
                out_dir_name = strip_part_from_base(out_dir_name, remove)
        else:
            if removes != "":
                out_dir_name = strip_part_from_base(out_dir_name, removes)

    # remove "part" naming
    out_dir_name = re.sub("\s*Pt(\.)?\s*\d\s*", "", out_dir_name)
    out_dir_name = re.sub("\s*Part\s*\d*", "", out_dir_name)
    out_dir_name = re.sub("\/\s*\d+\s*", "", out_dir_name)
    out_dir_name = out_dir_name.strip(" ()")

    # remove file dimensions
    out_dir_name = re.sub("(\[)?\d+x\d+(\])?", "", out_dir_name)

    # remove special case exceptions
    for exception, replace in file_name_exceptions.items():
        if out_dir_name == exception:
            out_dir_name = replace
            break

    for exception, replace in replace_exceptions.items():
        if exception in out_dir_name:
            out_dir_name = out_dir_name.replace(exception, replace)

    # remove numbers at the start and end of the name
    out_dir_name = re.sub("^#?\d{0,2}\s*", "", out_dir_name)
    out_dir_name = re.sub("#?\d{0,2}\s*$", "", out_dir_name)

    # clean up directory sizes
    # out_dir_name = re.sub("#?\d{2}\s*", "", out_dir_name)

    # clean up special characters at the start and end
    out_dir_name = re.sub("^\.\s*", "", out_dir_name)
    out_dir_name = re.sub("\s*\.$", "", out_dir_name)

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

    return out_dir_name


def clean_path(base_name, full_path):
    out_dir_name = base_name
    source_dir = Path(full_path)
    parts = source_dir.parts

    if base_name == "Base":
        return base_name

    # process duplicate entries in the path
    for part in parts:
        if part in parts:
            out_dir_name = strip_part_from_base(out_dir_name, part)

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

    return out_dir_name


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
        if base_name not in names_to_process:
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
                if computed_token not in GROUPING_EXCEPTIONS:
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
        subpath.name for subpath in folder.contents.values() if subpath.is_dir()
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

    for subfile in virtual_fs.contents.values():
        if subfile.is_dir():
            sub_name = subfile.name
            sub_paths = names_to_replace[sub_name]

            cleaned_name = os.path.join(*sub_paths)

            subfile.name = cleaned_name
            organize_groups(subfile)

    return virtual_fs
