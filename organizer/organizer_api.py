from fastapi import Depends, FastAPI, HTTPException, BackgroundTasks
from pathlib import Path
import json
import shutil
from datetime import datetime
from typing import Dict

from sqlalchemy import Cast, String, select
from api.models import GatherRequest, AsyncTaskResponse
from api.tasks import TaskInfo, tasks, update_task, TaskStatus, create_task
from data_models.database import (
    FolderStructure,
    get_session,
    GroupCategory,
    GroupCategoryEntry,
    setup_gather,
)

from api.api import (
    Category as CategoryAPI,
    CategoryResponse,
    FolderV2,
    SortColumn,
    SortOrder,
    StructureType,
    FolderViewResponse,
)
from sqlalchemy.orm import aliased
from sqlalchemy.sql import func
from fastapi.middleware.cors import CORSMiddleware

from pipeline.gather import gather_folder_structure_and_store
from grouping.group import group_folders
from pipeline.categorize import calculate_folder_structure

app = FastAPI()
app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]  # ty bug: FastAPI accepts middleware classes directly
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_path = "outputs/latest/latest.db"
output_dir = "outputs"


def get_db_session():
    return get_session(Path(db_path))


def get_folder_structure_from_db(
    db_path_str: str, stage: StructureType = StructureType.organized
) -> dict | None:
    """Get the latest folder structure from the database"""
    try:
        session = get_session(Path(db_path_str))
        newest_entry = session.execute(
            select(FolderStructure)
            .where(FolderStructure.structure_type == stage)
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
    sort_column: SortColumn = SortColumn.NAME,
    sort_order: SortOrder = SortOrder.asc,
    db=Depends(get_db_session),
) -> CategoryResponse:
    """
    Get the pre-calculated grouping with pagination
    """
    CategoryEntry = aliased(GroupCategoryEntry)

    offset = (page - 1) * page_size

    sort_column_to_attr = {
        SortColumn.NAME: GroupCategory.name,
        SortColumn.COUNT: GroupCategory.count,
        SortColumn.CONFIDENCE: GroupCategory.group_confidence,
        SortColumn.ID: GroupCategory.id,
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
        .where(FolderStructure.structure_type == StructureType.organized)
        .order_by(FolderStructure.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    entry = newest_entry.structure
    parsed_new_entry = json.loads(entry)

    old_entry = session.execute(
        select(FolderStructure)
        .where(FolderStructure.structure_type == StructureType.original)
        .order_by(FolderStructure.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    parsed_old_entry = json.loads(old_entry.structure)

    # Sort both folder structures by folder name
    sorted_original = sort_folder_structure(parsed_old_entry)
    sorted_new = sort_folder_structure(parsed_new_entry)

    return FolderViewResponse(
        original=FolderV2.model_validate(sorted_original),
        new=FolderV2.model_validate(sorted_new),
    )


@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str) -> TaskInfo:
    """Get the status of a task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return tasks[task_id]


@app.get("/api/tasks")
async def get_all_tasks() -> Dict[str, TaskInfo]:
    """Get all tasks (for debugging)"""
    return tasks


@app.get("/api/gather/structure")
async def get_gather_structure():
    """Get the original folder structure from the gather stage"""
    structure = get_folder_structure_from_db(db_path, StructureType.original)
    if structure is None:
        raise HTTPException(status_code=404, detail="No gather structure found")
    return {"folder_structure": structure}


@app.get("/api/group/structure")
async def get_group_structure():
    """Get the folder structure after grouping stage"""
    structure = get_folder_structure_from_db(db_path, StructureType.grouped)
    if structure is None:
        raise HTTPException(status_code=404, detail="No group structure found")
    return {"folder_structure": structure}


@app.get("/api/folders/structure")
async def get_folders_structure():
    """Get the final organized folder structure"""
    structure = get_folder_structure_from_db(db_path, StructureType.organized)
    if structure is None:
        raise HTTPException(status_code=404, detail="No organized structure found")
    return {"folder_structure": structure}


def run_gather_task(task_id: str, base_path_str: str):
    """Background task to run gather command"""
    try:
        update_task(
            task_id,
            status=TaskStatus.RUNNING,
            message="Starting gather process",
            progress=0.1,
        )

        base_path = Path(base_path_str)
        output_dir_path = Path(output_dir)

        # Validate paths
        if not base_path.exists() or not base_path.is_dir():
            update_task(
                task_id,
                status=TaskStatus.FAILED,
                error=f"Base path does not exist or is not a directory: {base_path}",
            )
            return

        update_task(task_id, message="Creating directories", progress=0.2)

        # Create output directory if it doesn't exist
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Create timestamped run directory
        timestamp_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = output_dir_path / timestamp_str
        run_dir.mkdir()

        update_task(task_id, message="Setting up database", progress=0.3)

        # Set up database
        db_path = run_dir / "run_data.db"
        setup_gather(db_path)

        update_task(task_id, message="Gathering folder structure", progress=0.4)

        # Gather folder structure
        gather_folder_structure_and_store(base_path, db_path)

        update_task(task_id, message="Post-processing filenames", progress=0.7)

        # Update latest symlinks
        latest_dir = output_dir_path / "latest"
        latest_db = latest_dir / "latest.db"

        # Remove existing latest directory if it exists
        if latest_dir.exists():
            shutil.rmtree(latest_dir)

        # Create new latest directory
        latest_dir.mkdir()

        # Copy the current run's database to latest.db
        shutil.copy2(db_path, latest_db)

        result = {
            "message": "Gather complete",
            "db_path": str(db_path),
            "run_dir": str(run_dir),
            "folder_structure": get_folder_structure_from_db(
                str(db_path), StructureType.original
            ),
        }

        update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            message="Gather complete",
            progress=1.0,
            result=result,
        )

    except Exception as e:
        update_task(task_id, status=TaskStatus.FAILED, error=str(e))


@app.post("/api/gather")
async def api_gather(
    request: GatherRequest, background_tasks: BackgroundTasks
) -> AsyncTaskResponse:
    """
    API endpoint version of the gather command (async).
    Returns a task ID that can be used to track progress.
    """
    task_id = create_task("Gather task created")

    # Start the background task
    background_tasks.add_task(run_gather_task, task_id, request.base_path)

    return AsyncTaskResponse(
        task_id=task_id, message="Gather task started", status=TaskStatus.PENDING
    )


def run_group_task(task_id: str):
    """Background task to run group command"""
    try:
        update_task(
            task_id,
            status=TaskStatus.RUNNING,
            message="Starting grouping process",
            progress=0.1,
        )

        db_path_obj = Path(db_path)

        # Validate database path
        if not db_path_obj.exists():
            update_task(
                task_id,
                status=TaskStatus.FAILED,
                error=f"Database path does not exist: {db_path}",
            )
            return

        update_task(task_id, message="Classifying folders", progress=0.2)

        # update_task(task_id, message="Grouping folders", progress=0.5)

        # Run grouping
        group_folders(db_path_obj)

        update_task(task_id, message="Calculating categories", progress=0.8)

        calculate_folder_structure(Path(db_path), structure_type=StructureType.grouped)

        update_task(task_id, message="Getting folder structure", progress=0.9)

        # Get folder structure if available
        folder_structure = get_folder_structure_from_db(
            db_path, stage=StructureType.grouped
        )

        result = {"message": "Grouping complete", "folder_structure": folder_structure}

        update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            message="Grouping complete",
            progress=1.0,
            result=result,
        )

    except Exception as e:
        update_task(task_id, status=TaskStatus.FAILED, error=str(e))


@app.post("/api/group")
async def api_group(background_tasks: BackgroundTasks) -> AsyncTaskResponse:
    """
    API endpoint version of the group command (async).
    Returns a task ID that can be used to track progress.
    """
    task_id = create_task("Group task created")

    # Start the background task
    background_tasks.add_task(run_group_task, task_id)

    return AsyncTaskResponse(
        task_id=task_id, message="Group task started", status=TaskStatus.PENDING
    )


def run_folders_task(task_id: str):
    """Background task to run folders command"""

    """
    Right now does nothing. Eventually will do the last stage of 
    categorizing and organizing
    """

    try:
        update_task(
            task_id,
            status=TaskStatus.RUNNING,
            message="Starting folder hierarchy generation",
            progress=0.1,
        )

        # Validate database path
        if not Path(db_path).exists():
            update_task(
                task_id,
                status=TaskStatus.FAILED,
                error=f"Database path does not exist: {db_path}",
            )
            return

        update_task(task_id, message="Calculating categories", progress=0.3)

        # Calculate categories and generate folder hierarchy
        calculate_folder_structure(
            Path(db_path), structure_type=StructureType.organized
        )

        # Get the newly generated folder structure
        folder_structure = get_folder_structure_from_db(
            db_path, stage=StructureType.organized
        )

        result = {
            "message": "Folder hierarchy generation complete",
            "folder_structure": folder_structure,
        }

        update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            message="Folder hierarchy generation complete",
            progress=1.0,
            result=result,
        )

    except Exception as e:
        update_task(task_id, status=TaskStatus.FAILED, error=str(e))


@app.post("/api/folders")
async def api_folders(background_tasks: BackgroundTasks) -> AsyncTaskResponse:
    """
    API endpoint version of the folders command (async).
    Returns a task ID that can be used to track progress.
    """
    task_id = create_task("Folders task created")

    # Start the background task
    background_tasks.add_task(run_folders_task, task_id)

    return AsyncTaskResponse(
        task_id=task_id, message="Folders task started", status=TaskStatus.PENDING
    )
