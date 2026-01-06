"""
Responsible for post-grouping processing

Apply categories at the file level,
combine groups that were over-zealously split
generate folder paths based on that

"""

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session
from storage.index_models import Node
from storage.work_models import GroupCategoryEntry

logger = logging.getLogger(__name__)


def get_categories_for_node(
    index_session: Session,
    work_session: Session,
    node: Node,
    iteration_id: int,
) -> list[GroupCategoryEntry]:
    """
    recursively get categories for the provided path
    """
    parent = index_session.execute(
        select(Node).where(Node.id == node.parent_node_id)
    ).scalar_one_or_none()

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

    categories = get_categories_for_node(
        index_session, work_session, parent, iteration_id
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
