from logging import getLogger
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any
from pydantic import BaseModel

logger = getLogger(__name__)

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    message: str
    progress: float = 0.0  # 0.0 to 1.0
    result: Dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


# In-memory task store (in production, use Redis or database)
tasks: Dict[str, TaskInfo] = {}


def create_task(message: str) -> str:
    """Create a new task and return its ID"""
    task_id = str(uuid.uuid4())
    now = datetime.now()

    task = TaskInfo(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message=message,
        created_at=now,
        updated_at=now,
    )

    tasks[task_id] = task
    logger.info(task)
    return task_id


def update_task(
    task_id: str,
    status: TaskStatus | None = None,
    message: str | None = None,
    progress: float | None = None,
    result: Dict[str, Any] | None = None,
    error: str | None = None,
):
    """Update an existing task"""
    
    if task_id not in tasks:
        logger.warning(f"Update for task {task_id} requested, but that task doesn't exist")
        return

    task = tasks[task_id]
    if status is not None:
        task.status = status
    if message is not None:
        task.message = message
    if progress is not None:
        task.progress = progress
    if result is not None:
        task.result = result
    if error is not None:
        task.error = error

    task.updated_at = datetime.now()
