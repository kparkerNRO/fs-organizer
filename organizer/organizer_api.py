from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import shutil
from typing import Optional
import json

from sqlalchemy import Cast, String, text
from pipeline.gather import gather_folder_structure_and_store, clean_file_name_post
from pipeline.classify import classify_folders
from grouping.group import categorize
from data_models.database import get_session, GroupCategory, GroupCategoryEntry

from data_models.api import (
    Folder as FolderAPI,
    Category as CategoryAPI,
    CategoryResponse,
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


class GatherRequest(BaseModel):
    base_path: str
    output_dir: str


@app.post("/gather")
async def gather_endpoint(request: GatherRequest):
    """
    Gather folder structure data and store in database
    """
    try:
        base_path = Path(request.base_path)
        output_dir = Path(request.output_dir)

        if not base_path.exists() or not base_path.is_dir():
            raise HTTPException(status_code=400, detail="Invalid base path")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped run directory
        timestamp_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = output_dir / timestamp_str
        run_dir.mkdir()

        # Set up and populate database
        db_path = run_dir / "run_data.db"
        # setup_gather(db_path)
        gather_folder_structure_and_store(base_path, db_path)

        # Handle latest symlink
        latest_dir = output_dir / "latest"
        latest_db = latest_dir / "latest.db"
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        latest_dir.mkdir()
        shutil.copy2(db_path, latest_db)

        return {
            "status": "success",
            "run_dir": str(run_dir),
            "db_path": str(db_path),
            "latest_db": str(latest_db),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/groups")
async def get_groups(
    page: int = 1, page_size: int = 10, db=Depends(get_db_session)
) -> CategoryResponse:
    """
    Get the pre-calculated grouping with pagination
    """
    CategoryEntry = aliased(GroupCategoryEntry)

    offset = (page - 1) * page_size

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
                    func.coalesce(CategoryEntry.new_name, "-"),
                    "original_name",
                    CategoryEntry.original_name,
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
        .join(CategoryEntry, CategoryEntry.group_id == GroupCategory.id)
        .filter(GroupCategory.group_confidence < 1.0)
        .filter(CategoryEntry.confidence > 0.0)
        .group_by(GroupCategory.id)
        .order_by(GroupCategory.id)
        .offset(offset)
        .limit(page_size)
    )
    result = db.execute(query).mappings().fetchall()
    results = [dict(row) for row in result]

    total_items_query = (
        db.query(func.count(func.distinct(GroupCategory.id)))
        .join(CategoryEntry, CategoryEntry.group_id == GroupCategory.id)
        .filter(GroupCategory.group_confidence < 1.0)
        .filter(CategoryEntry.confidence > 0.0)
    )
    total_items = db.execute(total_items_query).scalar()
    total_pages = (total_items + page_size - 1) // page_size

    categories = []
    for row in results:
        row["children"] = json.loads(row["children"])
        categories.append(CategoryAPI(**row))

    return CategoryResponse(
        data=categories,
        totalPages=total_pages,
        currentPage=page,
    )
