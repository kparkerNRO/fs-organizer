from fastapi import Depends, FastAPI, HTTPException
from pathlib import Path
import json
import shutil
from datetime import datetime

from sqlalchemy import Cast, String, select
from data_models.database import (
    FolderStructure,
    get_session,
    GroupCategory,
    GroupCategoryEntry,
    setup_gather,
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
from pydantic import BaseModel

from pipeline.gather import gather_folder_structure_and_store, clean_file_name_post
from pipeline.classify import classify_folders
from grouping.group import group_folders
from pipeline.categorize import calculate_categories
from pipeline.folder_reconstruction import generate_folder_heirarchy

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_path = "outputs/latest/latest.db"


class GatherRequest(BaseModel):
    base_path: str
    output_dir: str


class ProcessRequest(BaseModel):
    db_path: str


class GatherResponse(BaseModel):
    message: str
    db_path: str
    run_dir: str
    folder_structure: dict | None = None


class ProcessResponse(BaseModel):
    message: str
    folder_structure: dict | None = None


def get_db_session():
    return get_session(Path(db_path))


def get_folder_structure_from_db(db_path_str: str) -> dict | None:
    """Get the latest folder structure from the database"""
    try:
        session = get_session(Path(db_path_str))
        newest_entry = session.execute(
            select(FolderStructure)
            .where(FolderStructure.structure_type == StructureType.new)
            .order_by(FolderStructure.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        
        if newest_entry:
            parsed_entry = json.loads(newest_entry.structure)
            return sort_folder_structure(parsed_entry)
        return None
    except Exception:
        return None


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


@app.post("/api/gather")
async def api_gather(request: GatherRequest) -> GatherResponse:
    """
    API endpoint version of the gather command.
    1) Create a timestamped subfolder in output_dir,
    2) Create a run_data.db,
    3) Gather folder/file data,
    4) Insert freq counts.
    """
    try:
        base_path = Path(request.base_path)
        output_dir = Path(request.output_dir)
        
        # Validate paths
        if not base_path.exists() or not base_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Base path does not exist or is not a directory: {base_path}")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped run directory
        timestamp_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = output_dir / timestamp_str
        run_dir.mkdir()
        
        # Set up database
        db_path = run_dir / "run_data.db"
        setup_gather(db_path)
        
        # Gather folder structure
        gather_folder_structure_and_store(base_path, db_path)
        clean_file_name_post(db_path)
        
        # Update latest symlinks
        latest_dir = output_dir / "latest"
        latest_db = latest_dir / "latest.db"
        
        # Remove existing latest directory if it exists
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        
        # Create new latest directory
        latest_dir.mkdir()
        
        # Copy the current run's database to latest.db
        shutil.copy2(db_path, latest_db)
        
        return GatherResponse(
            message="Gather complete",
            db_path=str(db_path),
            run_dir=str(run_dir),
            folder_structure=None  # Gather doesn't generate folder structure yet
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during gather: {str(e)}")


@app.post("/api/group")
async def api_group(request: ProcessRequest) -> ProcessResponse:
    """
    API endpoint version of the group command.
    Classify and group folders in the given database.
    """
    try:
        db_path = Path(request.db_path)
        
        # Validate database path
        if not db_path.exists():
            raise HTTPException(status_code=400, detail=f"Database path does not exist: {db_path}")
        
        # Run classification first
        classify_folders(db_path)
        
        # Run grouping
        group_folders(db_path)
        calculate_categories(str(db_path))
        
        # Get folder structure if available
        folder_structure = get_folder_structure_from_db(str(db_path))
        
        return ProcessResponse(
            message="Grouping complete",
            folder_structure=folder_structure
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during grouping: {str(e)}")


@app.post("/api/folders")
async def api_folders(request: ProcessRequest) -> ProcessResponse:
    """
    API endpoint version of the folders command.
    Generate a folder hierarchy from the cleaned paths in the database.
    """
    try:
        db_path = request.db_path
        
        # Validate database path
        if not Path(db_path).exists():
            raise HTTPException(status_code=400, detail=f"Database path does not exist: {db_path}")
        
        # Calculate categories and generate folder hierarchy
        calculate_categories(db_path)
        generate_folder_heirarchy(db_path, type=StructureType.new)
        
        # Get the newly generated folder structure
        folder_structure = get_folder_structure_from_db(db_path)
        
        return ProcessResponse(
            message="Folder hierarchy generation complete",
            folder_structure=folder_structure
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during folder hierarchy generation: {str(e)}")
