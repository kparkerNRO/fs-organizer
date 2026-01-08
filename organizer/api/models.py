from enum import Enum
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel

from api.tasks import TaskStatus
from data_models.pipeline import Category
from data_models.pipeline import FolderV2


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
    data: List[Category] = []
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


# Dual Representation Models (for v2 API)


class HierarchyItem(BaseModel):
    """Represents either a file/directory from the filesystem (Node) or a semantic category."""

    id: str  # e.g., "node-123", "category-456"
    name: str
    type: Literal["node", "category"]
    originalPath: Optional[str] = None  # For nodes
    # Fields for compatibility with FolderV2/File
    confidence: float = 1.0  # Confidence score for classification
    possibleClassifications: list[str] = []  # Possible classifications for this item
    newPath: Optional[str] = None  # New path for files after organization
    count: int = 0  # Number of children (for folders/categories)


class DualRepresentation(BaseModel):
    """The complete data structure sent from the backend containing dual hierarchies."""

    items: Dict[str, HierarchyItem]
    node_hierarchy: Dict[str, List[str]]
    category_hierarchy: Dict[str, List[str]]


class HierarchyDiff(BaseModel):
    """Represents changes made by the user on the frontend (moving nodes between categories)."""

    added: Dict[str, List[str]]  # Key: Parent ID, Value: Child IDs that were added
    deleted: Dict[str, List[str]]  # Key: Parent ID, Value: Child IDs that were removed
