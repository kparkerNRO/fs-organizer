import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from api.api import StructureType
from sqlalchemy import func, select
from storage.index_models import Node
from storage.manager import StorageManager
from storage.work_models import FileMapping, GroupCategoryEntry
from utils.filename_processing import clean_filename
from utils.folder_structure import get_newest_entry_for_structure

from stages.categorize import get_categories_for_path

logger = logging.getLogger(__name__)


def _generate_folder_heirarchy_from_path(
    path: str, working_representation: Optional[Dict] = None
) -> dict:
    if working_representation is None:
        working_representation = {}
    cleaned_path = path
    if not cleaned_path:
        return working_representation
    cleaned_parts = cleaned_path.split("/")

    current_representation = working_representation
    for part in cleaned_parts:
        if part not in current_representation:
            current_representation[part] = {}
        current_representation = current_representation[part]
    if "__count__" not in current_representation:
        current_representation["__count__"] = 0
    current_representation["__count__"] += 1

    return working_representation


def get_folder_heirarchy(manager: StorageManager, run_id: int, structure_type: StructureType):
    logger.debug(f"Current working directory: {os.getcwd()}")

    with manager.get_work_session() as session:
        entry = get_newest_entry_for_structure(session, structure_type, run_id)
        if entry is not None:
            entry = json.loads(entry)
            pretty_entry = json.dumps(entry, indent=4)
            logger.info(pretty_entry)
        return entry


def _resolve_cleaned_name(node: Node) -> str:
    base_name = str(node.name)
    return clean_filename(base_name)


def _build_cleaned_path(
    node: Node,
    node_by_abs_path: Dict[str, Node],
    cache: Dict[int, str],
) -> str:
    if node.id in cache:
        return cache[node.id]

    name = _resolve_cleaned_name(node)
    parent_path_str = str(node.parent) if node.parent else None

    if not parent_path_str:
        cleaned_path = name
    else:
        parent = node_by_abs_path.get(parent_path_str)
        if parent is None:
            cleaned_path = name
        else:
            parent_cleaned = _build_cleaned_path(parent, node_by_abs_path, cache)
            if parent_cleaned:
                cleaned_path = str(Path(parent_cleaned) / name)
            else:
                cleaned_path = name

    cache[node.id] = cleaned_path
    return cleaned_path


def recalculate_cleaned_paths(manager: StorageManager, snapshot_id: int, run_id: int) -> int:
    with (
        manager.get_index_session(read_only=True) as index_session,
        manager.get_work_session() as work_session,
    ):
        nodes = (
            index_session.execute(
                select(Node).where(Node.snapshot_id == snapshot_id, Node.kind == "dir")
            )
            .scalars()
            .all()
        )
        node_by_abs_path = {node.abs_path: node for node in nodes}
        cache: Dict[int, str] = {}

        mappings = []
        for node in nodes:
            cleaned_path = _build_cleaned_path(node, node_by_abs_path, cache)

            # Check if a mapping already exists
            existing_mapping = work_session.execute(
                select(FileMapping).where(
                    FileMapping.run_id == run_id, FileMapping.node_id == node.id
                )
            ).scalar_one_or_none()

            if existing_mapping:
                existing_mapping.new_path = cleaned_path
            else:
                mappings.append(
                    FileMapping(
                        run_id=run_id,
                        node_id=node.id,
                        original_path=node.abs_path,
                        new_path=cleaned_path,
                    )
                )

        if mappings:
            work_session.add_all(mappings)
        work_session.commit()

    return len(nodes)


def recalculate_cleaned_paths_for_structure(
    manager: StorageManager,
    snapshot_id: int,
    run_id: int,
    structure_type: StructureType,
) -> int:
    if structure_type == StructureType.original:
        return recalculate_cleaned_paths(manager, snapshot_id, run_id)

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

        for node in nodes:
            categories = get_categories_for_path(
                index_session=index_session,
                work_session=work_session,
                node=node,
                iteration_id=iteration_id,
                snapshot_id=snapshot_id
            )
            cleaned_path = "/".join([str(category.processed_name) for category in categories])

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
