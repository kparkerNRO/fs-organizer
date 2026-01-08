from enum import Enum
from typing import Any, Dict, List, Literal, Optional
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


class HierarchyRecord(BaseModel):
    """
    Represents a node in a hierarchical tree structure.

    This is a recursive structure where each record can have children.
    Metadata dict is provided for future extensibility.
    """

    id: str  # Item ID (e.g., "node-123", "category-456")
    children: List["HierarchyRecord"] = []  # Child records
    metadata: Dict[str, Any] = {}  # Extensible metadata for future use


class Hierarchy(BaseModel):
    """
    Represents a single hierarchy for a specific pipeline stage.

    A hierarchy defines parent-child relationships between items using a tree structure.
    The same items can appear in multiple hierarchies with different relationships.
    """

    stage: str  # Pipeline stage name (e.g., "original", "organized", "grouped")
    source_type: Literal["node", "category"]  # Database table this was built from
    root: HierarchyRecord  # Root of the hierarchy tree


class DualRepresentation(BaseModel):
    """
    The complete data structure containing items and multiple stage-based hierarchies.

    This design supports:
    - Multiple pipeline stages (original, organized, grouped, etc.)
    - Each stage having its own hierarchy structure
    - Stages can be built from different source types (nodes or categories)
    - Shared item pool across all hierarchies for efficiency
    """

    items: Dict[str, HierarchyItem]  # Shared pool of all items
    hierarchies: Dict[str, Hierarchy]  # stage_name -> Hierarchy


class HierarchyDiff(BaseModel):
    """Represents changes made by the user on the frontend (moving nodes between categories)."""

    added: Dict[str, List[str]]  # Key: Parent ID, Value: Child IDs that were added
    deleted: Dict[str, List[str]]  # Key: Parent ID, Value: Child IDs that were removed


# Rebuild model to handle forward references (recursive HierarchyRecord)
HierarchyRecord.model_rebuild()
