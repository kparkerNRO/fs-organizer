from pydantic import BaseModel

from api.tasks import TaskStatus


class GatherRequest(BaseModel):
    base_path: str


class ProcessRequest(BaseModel):
    pass  # No parameters needed - uses constant db_path


class AsyncTaskResponse(BaseModel):
    task_id: str
    message: str
    status: TaskStatus


class GatherResponse(BaseModel):
    message: str
    db_path: str
    run_dir: str
    folder_structure: dict | None = None


class ProcessResponse(BaseModel):
    message: str
    folder_structure: dict | None = None
