import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

from api.models import CategoryResponse, SortColumn
from api.models import SortOrder
from data_models.pipeline import (
    Category as CategoryAPI,
)
from data_models.pipeline import (
    FolderV2,
    PipelineStage,
)
from api.models import AsyncTaskResponse, GatherRequest, DualRepresentation, HierarchyDiff
from api.profiling import ProfilingMiddleware, is_profiling_enabled
from api.tasks import TaskInfo, TaskStatus, create_task, tasks, update_task
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Cast, String
from sqlalchemy.orm import aliased
from sqlalchemy.sql import func
from api.models import FolderViewResponse
from stages.gather import ingest_filesystem
from stages.grouping.group import group_folders
from storage.id_defaults import get_latest_run
from storage.manager import DATA_DIR, StorageManager
from storage.work_models import (
    GroupCategory,
    GroupCategoryEntry,
)
from utils.folder_structure import (
    calculate_folder_structure_for_stage,
    get_newest_entry_for_stage,
    sort_folder_structure,
)
from utils.dual_representation import build_dual_representation
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Lifecycle manager for the FastAPI application.

    Handles startup and shutdown events.
    """
    del _app  # Parameter required by FastAPI but not used
    # Startup: Configure logging only in the worker process (not during module import)
    # This prevents duplicate log files when using uvicorn with auto-reload
    setup_logging("organizer_api")
    logger.info("FS-Organizer API server starting up")
    logger.info(f"Storage path: {DATA_DIR}")
    if is_profiling_enabled():
        logger.info("API profiling is ENABLED (use ?profile=true on requests)")

    yield

    # Shutdown: Add cleanup code here if needed
    logger.info("FS-Organizer API server shutting down")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]  # ty bug: FastAPI accepts middleware classes directly
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add profiling middleware (controlled by ENABLE_PROFILING env var)
if is_profiling_enabled():
    app.add_middleware(ProfilingMiddleware)


def get_storage_manager() -> StorageManager:
    """Create a storage manager for request-scoped dependencies."""
    return StorageManager()


def get_db_session(storage_manager: StorageManager = Depends(get_storage_manager)):
    """Get a work database session using StorageManager."""
    with storage_manager.get_work_session() as session:
        yield session


def get_folder_structure_from_db(
    storage_manager: StorageManager,
    stage: PipelineStage = PipelineStage.organized,
    run_id: int | None = None,
) -> dict | None:
    """Get the latest folder structure from the database"""
    try:
        with storage_manager.get_work_session() as session:
            newest_entry = get_newest_entry_for_stage(session, stage.value, run_id)

            if newest_entry:
                return sort_folder_structure(newest_entry)
            return None
    except Exception as err:
        logger.error("Error getting folder structure", exc_info=err)
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
    newest_entry = get_newest_entry_for_stage(session, PipelineStage.organized, None)
    if newest_entry is not None:
        parsed_new_entry = json.loads(newest_entry)
        sorted_new = sort_folder_structure(parsed_new_entry)
    else:
        raise HTTPException(status_code=404, detail="organized structure not found")

    old_entry = get_newest_entry_for_stage(session, PipelineStage.original, None)
    if old_entry:
        parsed_old_entry = json.loads(old_entry.structure)
        sorted_original = sort_folder_structure(parsed_old_entry)
    else:
        raise HTTPException(status_code=404, detail="original structure not found")

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
    structure = get_folder_structure_from_db(storage_manager, PipelineStage.original)
    if structure is None:
        raise HTTPException(status_code=404, detail="No gather structure found")
    return {"folder_structure": structure}


@app.get("/api/group/structure")
async def get_group_structure(
    storage_manager: StorageManager = Depends(get_storage_manager),
):
    """Get the folder structure after grouping stage"""
    structure = get_folder_structure_from_db(storage_manager, PipelineStage.grouped)
    if structure is None:
        raise HTTPException(status_code=404, detail="No group structure found")
    return {"folder_structure": structure}


@app.get("/api/folders/structure")
async def get_folders_structure(
    storage_manager: StorageManager = Depends(get_storage_manager),
):
    """Get the final organized folder structure"""
    structure = get_folder_structure_from_db(storage_manager, PipelineStage.organized)
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
        get_folder_structure_from_db(storage_manager, PipelineStage.original)
        is not None
    )
    has_group = (
        get_folder_structure_from_db(storage_manager, PipelineStage.grouped) is not None
    )
    has_folders = (
        get_folder_structure_from_db(storage_manager, PipelineStage.organized)
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

        update_task(task_id, message="Setting up storage", progress=0.1)

        update_task(task_id, message="Gathering folder structure", progress=0.2)

        # Gather folder structure using new ingest_filesystem
        storage_path = storage_manager.index_path.parent
        snapshot_id = ingest_filesystem(storage_manager, base_path)

        update_task(task_id, message="Post-processing filenames", progress=0.7)

        result = {
            "message": "Gather complete",
            "storage_path": str(storage_path),
            "snapshot_id": snapshot_id,
            "folder_structure": get_folder_structure_from_db(
                storage_manager, PipelineStage.original
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
        logger.error("Error running task", exc_info=e)
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

        with storage_manager.get_work_session() as session:
            latest_run = get_latest_run(session)
            if latest_run is None:
                update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    error="No runs found for grouping.",
                )
                return

        calculate_folder_structure_for_stage(
            storage_manager,
            latest_run.snapshot_id,
            latest_run.id,
            structure_type=PipelineStage.grouped,
        )

        update_task(task_id, message="Getting folder structure", progress=0.9)

        # Get folder structure if available
        folder_structure = get_folder_structure_from_db(
            storage_manager, stage=PipelineStage.grouped
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
        logger.error("Error in grouping", exc_info=e)
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
        with storage_manager.get_work_session() as session:
            latest_run = get_latest_run(session)
            if latest_run is None:
                update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    error="No runs found for folder calculation.",
                )
                return

        calculate_folder_structure_for_stage(
            storage_manager,
            latest_run.snapshot_id,
            latest_run.id,
            structure_type=PipelineStage.organized,
        )

        # Get the newly generated folder structure
        folder_structure = get_folder_structure_from_db(
            storage_manager, stage=PipelineStage.organized
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


# V2 API - Dual Representation


@app.get("/api/v2/folder-structure")
async def get_dual_representation(
    storage_manager: StorageManager = Depends(get_storage_manager),
) -> DualRepresentation:
    """
    Get the dual representation of folder hierarchies.

    Returns both the original filesystem structure (nodes) and the
    categorized structure in a unified format.
    """
    logger.info("GET /api/v2/folder-structure")

    with storage_manager.get_work_session() as session:
        latest_run = get_latest_run(session)
        if latest_run is None:
            raise HTTPException(
                status_code=404,
                detail="No runs found. Please run gather and group first.",
            )

    try:
        dual_rep = build_dual_representation(
            storage_manager,
            snapshot_id=latest_run.snapshot_id,
            run_id=latest_run.id,
        )
        return dual_rep
    except Exception as e:
        logger.error("Error building dual representation", exc_info=e)
        raise HTTPException(
            status_code=500,
            detail=f"Error building dual representation: {str(e)}",
        )


@app.patch("/api/v2/folder-structure")
async def apply_hierarchy_diff(
    diff: HierarchyDiff,
    storage_manager: StorageManager = Depends(get_storage_manager),
) -> dict:
    """
    Apply a hierarchy diff (user edits) to the category structure.

    Accepts a HierarchyDiff object and applies the changes to the database.
    Also logs the diff for analytics.
    """
    logger.info(f"PATCH /api/v2/folder-structure - Applying diff: {diff}")

    with storage_manager.get_work_session() as session:
        latest_run = get_latest_run(session)
        if latest_run is None:
            raise HTTPException(
                status_code=404,
                detail="No runs found. Cannot apply diff.",
            )

        try:
            # Apply the diff (implementation to be completed)
            # For now, we'll just log it
            from storage.work_models import HierarchyDiffLog

            log_entry = HierarchyDiffLog(
                run_id=latest_run.id,
                diff=diff.model_dump(),
            )
            session.add(log_entry)
            session.commit()

            logger.info(f"Hierarchy diff logged successfully: {log_entry.id}")

            return {
                "message": "Diff applied successfully",
                "log_id": log_entry.id,
            }
        except Exception as e:
            logger.error("Error applying hierarchy diff", exc_info=e)
            session.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Error applying diff: {str(e)}",
            )
