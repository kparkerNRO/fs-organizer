import logging
from pathlib import Path
import re
from typing import Optional

from sqlalchemy import func, select

from organizer.api.api import PipelineStage
from organizer.storage.index_models import Node
from organizer.storage.manager import StorageManager
from organizer.storage.work_models import FileMapping, GroupCategoryEntry
from organizer.utils.folder_structure import get_groups_for_node
from utils.common import VIEW_TYPES
from utils.config import Config, get_config

PATH_EXTRAS = " -,()/"

logger = logging.getLogger(__name__)


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


def _clean_filename(
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


def clean_filename(
    name: str,
    config: Optional[Config] = None,
) -> str:
    if name.lower().endswith(".zip"):
        return _clean_filename(name[:-4], config)
    return _clean_filename(name, config)


########################################
# Path processing
#########################################


def _build_cleaned_path(
    node: Node,
    node_by_abs_path: dict[str, Node],
) -> str:
    name = clean_filename(node.name)
    parent_path_str = str(node.parent) if node.parent else None

    if not parent_path_str:
        cleaned_path = name
    else:
        parent = node_by_abs_path.get(parent_path_str)
        if parent is None:
            cleaned_path = name
        else:
            parent_cleaned = _build_cleaned_path(parent, node_by_abs_path)
            if parent_cleaned:
                cleaned_path = str(Path(parent_cleaned) / name)
            else:
                cleaned_path = name

    return cleaned_path


def calculate_cleaned_paths_from_groups(
    manager: StorageManager,
    snapshot_id: int,
    run_id: int,
    structure_type: PipelineStage,
) -> int:
    with (
        manager.get_index_session(read_only=True) as index_session,
        manager.get_work_session() as work_session,
    ):
        iteration_id = work_session.execute(
            select(func.max(GroupCategoryEntry.iteration_id))
        ).scalar_one()

        nodes = (
            index_session.execute(
                select(Node).where(Node.snapshot_id == snapshot_id, Node.kind == "dir")
            )
            .scalars()
            .all()
        )

        node_by_abs_path = {node.abs_path: node for node in nodes}
        for node in nodes:
            if structure_type == PipelineStage.original:
                cleaned_path = _build_cleaned_path(node, node_by_abs_path)

            else:
                categories = get_groups_for_node(
                    index_session=index_session,
                    work_session=work_session,
                    node=node,
                    iteration_id=iteration_id,
                )
                cleaned_path = "/".join(
                    [str(category.processed_name) for category in categories]
                )

            existing_mapping = work_session.execute(
                select(FileMapping).where(
                    FileMapping.run_id == run_id, FileMapping.node_id == node.id
                )
            ).scalar_one_or_none()

            if existing_mapping:
                existing_mapping.new_path = cleaned_path
            else:
                work_session.add(
                    FileMapping(
                        run_id=run_id,
                        node_id=node.id,
                        original_path=node.abs_path,
                        new_path=cleaned_path,
                    )
                )

        work_session.commit()

    return len(nodes)
