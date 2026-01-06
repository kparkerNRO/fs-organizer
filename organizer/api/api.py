from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from storage.index_models import Node


class StructureType(str, Enum):
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


class FolderViewResponse(BaseModel):
    original: FolderV2
    new: FolderV2


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
