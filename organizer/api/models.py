from enum import Enum

from pydantic import BaseModel

from api.tasks import TaskStatus
from data_models.pipeline import Category, FolderV2, Hierarchy, HierarchyItem


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


class CategoryResponse(BaseModel):
    data: list[Category] = []
    totalItems: int = 0
    totalPages: int = 1
    currentPage: int = 1


class SortColumn(str, Enum):
    NAME = "name"
    COUNT = "count"
    CONFIDENCE = "confidence"
    ID = "id"


class SortOrder(str, Enum):
    asc = "asc"
    desc = "desc"


class FolderViewResponse(BaseModel):
    original: FolderV2
    new: FolderV2


# Dual Representation API Models


class DualRepresentation(BaseModel):
    """
    API-level payload containing shared items and multiple hierarchies.

    This is only used at the API boundary. Internal functions work with
    individual Hierarchy objects.
    """

    items: dict[str, HierarchyItem]  # Shared pool (ItemStore)
    hierarchies: dict[str, Hierarchy]  # stage_name -> Hierarchy


class HierarchyDiff(BaseModel):
    """
    Lightweight object describing changes made to a Hierarchy.

    Analogous to a git diff or commit. Allows efficient transmission
    of hierarchy modifications without sending the entire tree.
    """

    added: dict[str, list[str]]  # Key: Parent ID, Value: Child IDs to add
    deleted: dict[str, list[str]]  # Key: Parent ID, Value: Child IDs to remove
