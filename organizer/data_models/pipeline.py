from enum import Enum
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field
from storage.index_models import Node


class PipelineStage(str, Enum):
    original = "old"
    organized = "new"
    grouped = "grouped"


class NodeType(str, Enum):
    file = "file"
    folder = "folder"


class FSNode(BaseModel):
    type: NodeType
    id: int | None = None
    name: str
    confidence: float = 1.0
    possibleClassifications: list[str] = []
    originalPath: str | None = None


class File(FSNode):
    type: Literal[NodeType.file] = Field(default=NodeType.file)
    originalPath: str
    newPath: str | None = None

    @staticmethod
    def from_node(node: Node):
        return File(id=node.id, name=node.name, originalPath=node.rel_path)


class FolderV2(FSNode):
    type: Literal[NodeType.folder] = Field(default=NodeType.folder)
    count: int = 0
    children: list["File | FolderV2"] = []

    @property
    def children_map(self):
        return {child.name: child for child in self.children}

    @staticmethod
    def from_node(node: Node):
        return FolderV2(id=node.id, name=node.name, originalPath=node.rel_path)


class Folder(BaseModel):
    id: int
    name: str
    classification: str | None = None
    original_filename: str | None = None
    cleaned_name: str | None = None
    confidence: float | None = None
    original_path: str
    processed_names: List[str] = []


class Category(BaseModel):
    id: int
    name: str
    classification: str | None = None
    count: int
    confidence: float
    possibleClassifications: Optional[List[str]] = None
    children: Optional[List[Folder]] = None


# Dual Representation Models (Git-like architecture)


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


# Rebuild model to handle forward references (recursive HierarchyRecord)
HierarchyRecord.model_rebuild()
