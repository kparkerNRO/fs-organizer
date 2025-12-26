import re
import logging
from typing import List, Optional

from utils.common import VIEW_TYPES
from utils.config import Config, get_config

PATH_EXTRAS = " -,()/"

logger = logging.getLogger(__name__)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def has_close_match(s: str, candidates: List[str], max_distance: int = 2) -> bool:
    """
    Check if string s has a Levenshtein distance <= max_distance with any string in candidates.

    Args:
        s: The string to check
        candidates: List of candidate strings to compare against
        max_distance: Maximum Levenshtein distance to consider a match (default: 2)

    Returns:
        True if any candidate has distance <= max_distance, False otherwise
    """
    return any(levenshtein_distance(s, candidate) <= max_distance for candidate in candidates)


def strip_part_from_base(base_name: str, part):
    """
    Removes a string "part" from the file name, removing any spaces or additional
    folders (eg, avoids returning <path>//<name>)
    """
    output = base_name.replace(part, "")
    output = re.sub(r"\s+", " ", output)
    output = re.sub(r"\/\/", "/", output)
    output = output.strip()

    return output


def get_max_common_string(tokens, name_to_comp):
    base_token = tokens[0]
    working_token = [base_token]
    lower_tokens = [token.lower() for token in tokens]
    lower_name = name_to_comp.lower()

    # greedily add tokens until they stop matching
    for i in range(1, len(tokens) + 1):
        test_token = " ".join(lower_tokens[0:i])
        if lower_name.startswith(test_token):
            working_token = lower_tokens[0:i]
        else:
            break

    shared_tokens = " ".join(tokens[: len(working_token)])
    return shared_tokens


def get_max_common_words(tokens, name_to_comp):
    base_token = tokens[0]
    working_token = [base_token]
    lower_tokens = [token.lower() for token in tokens]

    lower_name = name_to_comp.lower()
    lower_comp_tokens = lower_name.split(" ")

    # greedily add tokens until they stop matching
    for i in range(1, len(tokens) + 1):
        test_tokens = lower_tokens[0:i]
        lower_comp_test_tokens = lower_comp_tokens[0:i]
        if lower_comp_test_tokens == test_tokens:
            working_token = lower_tokens[0:i]
        else:
            break

    shared_tokens = " ".join(tokens[: len(working_token)])
    return shared_tokens


def split_view_type(base_name, view_types=VIEW_TYPES):
    """
    Trims out standard view type suffixes from a filename, ensuring
    that the view type is a full word, not part of another word.
    """
    f_suffix = None
    f_name = base_name
    for suffix in view_types:
        # Ensure the suffix is a full word using regex
        if re.search(rf"\b{re.escape(suffix)}\b", base_name):
            f_suffix = suffix.strip()
            f_name = strip_part_from_base(base_name, suffix)
            break

    return f_name, f_suffix


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
    base_name,
    config: Optional[Config] = None,
):
    """
    Handles a bunch of standardized cleanup for junk that ends up in folder names
    """
    config = config or get_config()
    if base_name in config.clean_exceptions:
        return base_name

    out_dir_name = base_name
    # remove myairbridge tags
    if "myairbridge" in base_name:
        out_dir_name = base_name.replace("myairbridge-", "")

    # convert underscores into spaces or 's'
    if "_" in base_name and base_name != "_Unsorted":
        if "_s" in base_name:
            out_dir_name = base_name.replace("_s", "s")

        else:
            out_dir_name = base_name.replace("_", " ")

    # remove any creator-specific removals
    for _, removes in config.creator_removes.items():
        for remove in removes:
            if remove:
                out_dir_name = strip_part_from_base(out_dir_name, remove)

    # remove "part" naming
    out_dir_name = re.sub(r"\s*Pt(\.)?\s*\d\s*", "", out_dir_name)
    out_dir_name = re.sub(r"\s*Part\s*\d*", "", out_dir_name)
    out_dir_name = re.sub(r"\/\s*\d+\s*", "", out_dir_name)
    out_dir_name = out_dir_name.strip(" ()")

    # remove file dimensions
    out_dir_name = re.sub(r"(\[)?\d+x\d+(\])?", "", out_dir_name)

    # remove special case exceptions
    for exception, replace in config.file_name_exceptions.items():
        if out_dir_name == exception:
            out_dir_name = replace
            break

    for exception, replace in config.replace_exceptions.items():
        if exception in out_dir_name:
            out_dir_name = out_dir_name.replace(exception, replace)

    # remove numbers at the start and end of the name
    out_dir_name = re.sub(r"^(♯|#?)\d{0,3}\s*", "", out_dir_name)
    out_dir_name = re.sub(r"#?\d{0,2}\s*$", "", out_dir_name)

    # remove "#<number> at the start of the name"
    out_dir_name = re.sub(r"^#\d+\s*", "", out_dir_name)

    # clean up special characters at the start and end
    out_dir_name = re.sub(r"^\.\s*", "", out_dir_name)
    out_dir_name = re.sub(r"\s*\.$", "", out_dir_name)

    # clean up hyphens *only* if there is no whitespace in the name
    if " " not in out_dir_name:
        out_dir_name = out_dir_name.replace("-", " ")

    # cleanup whitespace
    out_dir_name = out_dir_name.replace("(", " ")
    out_dir_name = re.sub(r"\s+", " ", out_dir_name)
    out_dir_name = re.sub(r"--", " ", out_dir_name)
    out_dir_name = re.sub(r"-\s+-", " ", out_dir_name)
    out_dir_name = out_dir_name.strip(PATH_EXTRAS)

    # cleanup number only entries
    out_dir_name = re.sub(r"^\s*\d+\s*$", "", out_dir_name)

    if len(out_dir_name) == 1:
        out_dir_name = ""

    return out_dir_name
