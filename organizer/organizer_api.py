from fastapi import Depends, FastAPI, HTTPException, BackgroundTasks
from pathlib import Path
import json
import logging
import sys
from typing import Dict

from sqlalchemy import Cast, String, select
from sqlalchemy.orm import aliased
from sqlalchemy.sql import func
from fastapi.middleware.cors import CORSMiddleware

from api.models import GatherRequest, AsyncTaskResponse
from api.tasks import TaskInfo, tasks, update_task, TaskStatus, create_task
from storage.work_models import (
    FolderStructure,
    GroupCategory,
    GroupCategoryEntry,
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
from stages.gather import ingest_filesystem
from stages.grouping.group import group_folders
from stages.categorize import calculate_folder_structure
from storage.id_defaults import get_latest_run
from storage.manager import StorageManager, DATA_DIR
from utils.folder_structure import sort_folder_structure

# Configure logging to INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]  # ty bug: FastAPI accepts middleware classes directly
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")  # type: ignore[deprecated]
async def startup_event():
    """Log when the API server starts"""
    logger.info("FS-Organizer API server starting up")
    logger.info(f"Storage path: {DATA_DIR}")


def get_storage_manager() -> StorageManager:
    """Create a storage manager for request-scoped dependencies."""
    return StorageManager()


def get_db_session(storage_manager: StorageManager = Depends(get_storage_manager)):
    """Get a work database session using StorageManager."""
    with storage_manager.get_work_session() as session:
        yield session


def get_folder_structure_from_db(
    storage_manager: StorageManager, stage: StructureType = StructureType.organized
) -> dict | None:
    """Get the latest folder structure from the database"""
    try:
        with storage_manager.get_work_session() as session:
            newest_entry = session.execute(
                select(FolderStructure)
                .where(FolderStructure.structure_type == stage.value)
                .order_by(FolderStructure.id.desc())
                .limit(1)
            ).scalar_one_or_none()

            if newest_entry:
                return sort_folder_structure(newest_entry.structure)
            return None
    except Exception:
        return None


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
    logger.info(
        f"GET /groups - page: {page}, page_size: {page_size}, sort: {sort_column}:{sort_order}"
    )
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
        .join(CategoryEntry, CategoryEntry.group_id == GroupCategory.id)
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
        .join(CategoryEntry, CategoryEntry.group_id == GroupCategory.id)
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
    logger.info("GET /folders")
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
async def get_gather_structure(
    storage_manager: StorageManager = Depends(get_storage_manager),
):
    """Get the original folder structure from the gather stage"""
    structure = get_folder_structure_from_db(storage_manager, StructureType.original)
    if structure is None:
        raise HTTPException(status_code=404, detail="No gather structure found")
    return {"folder_structure": structure}


@app.get("/api/group/structure")
async def get_group_structure(
    storage_manager: StorageManager = Depends(get_storage_manager),
):
    """Get the folder structure after grouping stage"""
    structure = get_folder_structure_from_db(storage_manager, StructureType.grouped)
    if structure is None:
        raise HTTPException(status_code=404, detail="No group structure found")
    return {"folder_structure": structure}


@app.get("/api/folders/structure")
async def get_folders_structure(
    storage_manager: StorageManager = Depends(get_storage_manager),
):
    """Get the final organized folder structure"""
    structure = get_folder_structure_from_db(storage_manager, StructureType.organized)
    if structure is None:
        raise HTTPException(status_code=404, detail="No organized structure found")
    return {"folder_structure": structure}


@app.get("/api/status")
async def get_pipeline_status(
    storage_manager: StorageManager = Depends(get_storage_manager),
):
    """Check which pipeline stages have available data"""
    logger.info("GET /api/status - checking pipeline stages")

    has_gather = (
        get_folder_structure_from_db(storage_manager, StructureType.original)
        is not None
    )
    has_group = (
        get_folder_structure_from_db(storage_manager, StructureType.grouped) is not None
    )
    has_folders = (
        get_folder_structure_from_db(storage_manager, StructureType.organized)
        is not None
    )

    return {
        "has_gather": has_gather,
        "has_group": has_group,
        "has_folders": has_folders,
        "storage_path": str(storage_manager.index_path.parent),
    }


def run_gather_task(
    task_id: str,
    base_path_str: str,
    storage_manager: StorageManager,
):
    """Background task to run gather command"""
    try:
        update_task(
            task_id,
            status=TaskStatus.RUNNING,
            message="Starting gather process",
            progress=0.1,
        )

        base_path = Path(base_path_str)

        # Validate paths
        if not base_path.exists() or not base_path.is_dir():
            update_task(
                task_id,
                status=TaskStatus.FAILED,
                error=f"Base path does not exist or is not a directory: {base_path}",
            )
            return

        update_task(task_id, message="Setting up storage", progress=0.3)

        update_task(task_id, message="Gathering folder structure", progress=0.4)

        # Gather folder structure using new ingest_filesystem
        storage_path = storage_manager.index_path.parent
        snapshot_id = ingest_filesystem(storage_manager, base_path)

        update_task(task_id, message="Post-processing filenames", progress=0.7)

        result = {
            "message": "Gather complete",
            "storage_path": str(storage_path),
            "snapshot_id": snapshot_id,
            "folder_structure": get_folder_structure_from_db(
                storage_manager, StructureType.original
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
    request: GatherRequest,
    background_tasks: BackgroundTasks,
    storage_manager: StorageManager = Depends(get_storage_manager),
) -> AsyncTaskResponse:
    """
    API endpoint version of the gather command (async).
    Returns a task ID that can be used to track progress.
    """
    logger.info(f"POST /api/gather - base_path: {request.base_path}")
    task_id = create_task("Gather task created")

    # Start the background task
    background_tasks.add_task(
        run_gather_task, task_id, request.base_path, storage_manager
    )

    logger.info(f"Gather task created with ID: {task_id}")
    return AsyncTaskResponse(
        task_id=task_id, message="Gather task started", status=TaskStatus.PENDING
    )


def run_group_task(task_id: str, storage_manager: StorageManager):
    """Background task to run group command"""
    try:
        update_task(
            task_id,
            status=TaskStatus.RUNNING,
            message="Starting grouping process",
            progress=0.1,
        )

        update_task(task_id, message="Classifying folders", progress=0.2)

        # update_task(task_id, message="Grouping folders", progress=0.5)

        # Run grouping
        group_folders(storage_manager)

        update_task(task_id, message="Calculating categories", progress=0.8)

        latest_run = get_latest_run(storage_manager)
        if latest_run is None:
            update_task(
                task_id,
                status=TaskStatus.FAILED,
                error="No runs found for grouping.",
            )
            return

        calculate_folder_structure(
            storage_manager,
            latest_run.snapshot_id,
            latest_run.id,
            structure_type=StructureType.grouped,
        )

        update_task(task_id, message="Getting folder structure", progress=0.9)

        # Get folder structure if available
        folder_structure = get_folder_structure_from_db(
            storage_manager, stage=StructureType.grouped
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
async def api_group(
    background_tasks: BackgroundTasks,
    storage_manager: StorageManager = Depends(get_storage_manager),
) -> AsyncTaskResponse:
    """
    API endpoint version of the group command (async).
    Returns a task ID that can be used to track progress.
    """
    logger.info("POST /api/group")
    task_id = create_task("Group task created")

    # Start the background task
    background_tasks.add_task(run_group_task, task_id, storage_manager)

    logger.info(f"Group task created with ID: {task_id}")
    return AsyncTaskResponse(
        task_id=task_id, message="Group task started", status=TaskStatus.PENDING
    )


def run_folders_task(task_id: str, storage_manager: StorageManager):
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

        update_task(task_id, message="Calculating categories", progress=0.3)

        # Calculate categories and generate folder hierarchy
        latest_run = get_latest_run(storage_manager)
        if latest_run is None:
            update_task(
                task_id,
                status=TaskStatus.FAILED,
                error="No runs found for folder calculation.",
            )
            return

        calculate_folder_structure(
            storage_manager,
            latest_run.snapshot_id,
            latest_run.id,
            structure_type=StructureType.organized,
        )

        # Get the newly generated folder structure
        folder_structure = get_folder_structure_from_db(
            storage_manager, stage=StructureType.organized
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
async def api_folders(
    background_tasks: BackgroundTasks,
    storage_manager: StorageManager = Depends(get_storage_manager),
) -> AsyncTaskResponse:
    """
    API endpoint version of the folders command (async).
    Returns a task ID that can be used to track progress.
    """
    logger.info("POST /api/folders")
    task_id = create_task("Folders task created")

    # Start the background task
    background_tasks.add_task(run_folders_task, task_id, storage_manager)

    logger.info(f"Folders task created with ID: {task_id}")
    return AsyncTaskResponse(
        task_id=task_id, message="Folders task started", status=TaskStatus.PENDING
    )
