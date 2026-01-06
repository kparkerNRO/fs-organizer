import logging
from pathlib import Path

from api.api import StructureType
from sqlalchemy import func, select
from storage.index_models import Node
from storage.manager import StorageManager
from storage.work_models import FileMapping, GroupCategoryEntry
from utils.filename_processing import clean_filename

from stages.categorize import get_categories_for_node

logger = logging.getLogger(__name__)


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


def calculate_cleaned_paths_for_structure(
    manager: StorageManager,
    snapshot_id: int,
    run_id: int,
    structure_type: StructureType,
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
            if structure_type == StructureType.original:
                cleaned_path = _build_cleaned_path(node, node_by_abs_path)

            else:
                categories = get_categories_for_node(
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
