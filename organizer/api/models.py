from enum import Enum
from typing import Any

from pydantic import BaseModel

from api.tasks import TaskStatus
from data_models.pipeline import Category, FolderV2, PipelineStage


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


# Dual Representation Models (for v2 API)


class ItemType(str, Enum):
    """Type of item in the hierarchy."""

    NODE = "node"
    CATEGORY = "category"


class HierarchyItem(BaseModel):
    """
    Intrinsic, shared data for an underlying file, folder, or category.

    Lives in the ItemStore. This data is considered context-independent.
    The same item can appear in multiple hierarchies with different names/contexts.
    """

    id: str  # Unique, persistent ID (e.g., "node-123", "category-456")
    type: ItemType
    originalPath: str | None = None  # Immutable property of a file node


class HierarchyRecord(BaseModel):
    """
    Represents an item's instance within a specific tree.

    This is a recursive structure where each record can have children.
    It holds context-dependent properties, like the item's name in that tree.
    """

    itemId: str  # Foreign key pointing to HierarchyItem in ItemStore
    name: str  # The name of this item in this tree (contextual, can be mutated)
    children: list["HierarchyRecord"] = []  # Child records
    metadata: dict[str, Any] = {}  # Tree-specific metadata


class Hierarchy(BaseModel):
    """
    A complete, self-contained hierarchy for a specific pipeline stage.

    Analogous to FolderV2 - represents one tree structure.
    Multiple hierarchies can reference the same items in the ItemStore.
    """

    contained_ids: set[int]  # Database record IDs contained in this hierarchy
    structure_id: int  # ID of FolderStructure record if saved to DB
    run_id: int | None  # Associated run ID
    stage: PipelineStage  # Pipeline stage (e.g., ORIGINAL, ORGANIZED)
    source_type: ItemType  # Database table this was built from (NODE or CATEGORY)
    root: HierarchyRecord  # Root of the hierarchy tree


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


# Rebuild model to handle forward references (recursive HierarchyRecord)
HierarchyRecord.model_rebuild()
