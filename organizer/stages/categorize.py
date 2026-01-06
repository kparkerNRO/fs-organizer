"""
Responsible for post-grouping processing

Apply categories at the file level,
combine groups that were over-zealously split
generate folder paths based on that

"""

import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session
from storage.index_models import Node
from storage.work_models import GroupCategoryEntry

logger = logging.getLogger(__name__)


def get_parent_folder(
    session: Session, parent_path: Path, snapshot_id, zip_content: bool = False
) -> Node | None:
    """Find the parent folder entry based on its path."""

    parent_path_str = str(parent_path)
    parent = session.execute(
        select(Node).where(
            Node.abs_path == parent_path_str,
            Node.kind == "dir",
            Node.snapshot_id == snapshot_id,
        )
    ).scalar_one_or_none()

    if not parent and zip_content:
        parent_path = parent_path.parent
        parent = session.execute(
            select(Node).where(
                Node.abs_path == str(parent_path),
                Node.kind == "dir",
            )
        ).scalar_one_or_none()

    return parent


def get_categories_for_path(
    index_session: Session,
    work_session: Session,
    node: Node,
    # path: str | Path,
    iteration_id: int,
    snapshot_id: int,
) -> list[GroupCategoryEntry]:
    """
    recursively get categories for the provided path
    """
    # path = node.abs_path
    parent = index_session.execute(select(Node).where(Node.id == node.parent_node_id)).scalar_one_or_none()

    # parent_path = path.parent
    # zip_content = parent_path.match("*.zip")
    # parent = get_parent_folder(index_session, parent_path, snapshot_id, zip_content)

    if not parent:
        return []

    # GroupCategoryEntry uses folder_id which maps to node_id in the new schema
    groups = (
        work_session.execute(
            select(GroupCategoryEntry)
            .where(GroupCategoryEntry.iteration_id == iteration_id)
            .where(GroupCategoryEntry.folder_id == parent.id)
        )
        .scalars()
        .all()
    )

    categories = get_categories_for_path(
        index_session, work_session, parent, iteration_id, snapshot_id
    )
    category_names = {cat.processed_name: index for index, cat in enumerate(categories)}
    for group in groups:
        processed_name = group.processed_name
        if group.processed_name in category_names:
            categories[category_names[processed_name]].confidence = min(
                group.confidence, categories[category_names[processed_name]].confidence
            )
        else:
            categories.append(group)

    return categories
