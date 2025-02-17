from fastapi import Depends, FastAPI
from pathlib import Path
import json

from sqlalchemy import Cast, String
from data_models.database import get_session, GroupCategory, GroupCategoryEntry

from data_models.api import (
    Category as CategoryAPI,
    CategoryResponse,
    SortColumn,
    SortOrder,
)
from sqlalchemy.orm import aliased
from sqlalchemy.sql import func
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_path = "outputs/latest/latest.db"


def get_db_session():
    return get_session(Path(db_path))


@app.get("/groups")
async def get_groups(
    page: int = 1,
    page_size: int = 10,
    sort_column: SortColumn = SortColumn.name,
    sort_order: SortOrder = SortOrder.asc,
    db=Depends(get_db_session),
) -> CategoryResponse:
    """
    Get the pre-calculated grouping with pagination
    """
    CategoryEntry = aliased(GroupCategoryEntry)

    offset = (page - 1) * page_size

    sort_column_to_attr = {
        SortColumn.name: GroupCategory.name,
        SortColumn.count: GroupCategory.count,
        SortColumn.confidence: GroupCategory.group_confidence,
        SortColumn.id: GroupCategory.id,
    }

    sort_attr = sort_column_to_attr[sort_column]
    if sort_order == SortOrder.desc:
        sort_attr = sort_attr.desc()

    query = (
        db.query(
            GroupCategory.id.label("id"),
            GroupCategory.name.label("name"),
            GroupCategory.name.label("original_name"),
            GroupCategory.count.label("count"),
            GroupCategory.group_confidence.label("confidence"),
            func.json_group_array(
                func.json_object(
                    "id",CategoryEntry.id,
                    "name",func.coalesce(CategoryEntry.new_name, "-"),
                    "original_filename",CategoryEntry.original_name,
                    "original_path",CategoryEntry.path,
                    "processed_names",func.json(
                        Cast(func.coalesce(CategoryEntry.derived_names, "[]"), String)
                    ),
                    "confidence",CategoryEntry.confidence,
                )
            ).label("children"),
        )
        .join(CategoryEntry, CategoryEntry.new_group_id == GroupCategory.id)
        .filter(GroupCategory.group_confidence < 1.0)
        .group_by(GroupCategory.id)
        .order_by(sort_attr)
        .offset(offset)
        .limit(page_size)
    )
    result = db.execute(query).mappings().fetchall()
    results = [dict(row) for row in result]

    total_items_query = (
        db.query(func.count(func.distinct(GroupCategory.id)))
        .join(CategoryEntry, CategoryEntry.new_group_id == GroupCategory.id)
        .filter(GroupCategory.group_confidence < 1.0)
    )
    total_items = db.execute(total_items_query).scalar()
    total_pages = (total_items + page_size - 1) // page_size

    categories = []
    for row in results:
        row["children"] = json.loads(row["children"])
        categories.append(CategoryAPI(**row))

    return CategoryResponse(
        data=categories,
        totalItems=total_items,
        totalPages=total_pages,
        currentPage=page,
    )
