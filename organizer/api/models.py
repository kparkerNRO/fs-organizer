from pydantic import BaseModel

from api.tasks import TaskStatus


class GatherRequest(BaseModel):
    base_path: str


class ProcessRequest(BaseModel):
    pass  # No parameters needed - uses default storage path


class AsyncTaskResponse(BaseModel):
    task_id: str
    message: str
    status: TaskStatus


class GatherResponse(BaseModel):
    message: str
    storage_path: str
    folder_structure: dict | None = None


class ProcessResponse(BaseModel):
    message: str
    folder_structure: dict | None = None
