from fastapi import Depends, FastAPI
from pathlib import Path
import json

from sqlalchemy import Cast, String, select
from data_models.database import (
    FolderStructure,
    get_session,
    GroupCategory,
    GroupCategoryEntry,
)

from data_models.api import (
    Category as CategoryAPI,
    CategoryResponse,
    SortColumn,
    SortOrder,
    StructureType,
    FolderViewResponse,
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


def sort_folder_structure(folder_data: dict) -> dict:
    """
    Recursively sort folder structure by folder/file names
    """
    if not isinstance(folder_data, dict):
        return folder_data

    # Create a new FolderV2 object to ensure proper structure
    if "name" in folder_data and "children" in folder_data:
        # This is a folder object
        sorted_children = []

        # Sort children by name
        children = folder_data.get("children", [])
        if children:
            # Separate files and folders
            files = [child for child in children if "id" in child]
            folders = [child for child in children if "id" not in child]

            # Sort files by name
            files.sort(key=lambda x: x.get("name", "").lower())

            # Sort folders by name and recursively sort their children
            folders.sort(key=lambda x: x.get("name", "").lower())
            for folder in folders:
                sorted_children.append(sort_folder_structure(folder))

            # Add files after folders
            sorted_children.extend(files)

        # Return sorted folder
        return {**folder_data, "children": sorted_children}

    return folder_data


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
                    "id",
                    CategoryEntry.id,
                    "name",
                    func.coalesce(CategoryEntry.processed_name, "-"),
                    "original_filename",
                    CategoryEntry.pre_processed_name,
                    "original_path",
                    CategoryEntry.path,
                    "processed_names",
                    func.json(
                        Cast(func.coalesce(CategoryEntry.derived_names, "[]"), String)
                    ),
                    "confidence",
                    CategoryEntry.confidence,
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


@app.get("/folders")
async def get_folders(
    session=Depends(get_db_session),
):
    newest_entry = session.execute(
        select(FolderStructure)
        .where(FolderStructure.structure_type == StructureType.new)
        .order_by(FolderStructure.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    entry = newest_entry.structure
    parsed_new_entry = json.loads(entry)

    old_entry = session.execute(
        select(FolderStructure)
        .where(FolderStructure.structure_type == StructureType.old)
        .order_by(FolderStructure.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    parsed_old_entry = json.loads(old_entry.structure)

    # Sort both folder structures by folder name
    sorted_original = sort_folder_structure(parsed_old_entry)
    sorted_new = sort_folder_structure(parsed_new_entry)

    return FolderViewResponse(original=sorted_original, new=sorted_new)
