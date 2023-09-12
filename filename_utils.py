import re

from common import VIEW_TYPES

PATH_EXTRAS = " -,()/"
from pathlib import Path


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
