from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


class StructureType(str, Enum):
    original = "old"
    organized = "new"
    grouped = "grouped"


class File(BaseModel):
    id: int
    name: str
    confidence: float = 1.0
    possibleClassifications: list[str] = []
    originalPath: str
    newPath: str | None


class FolderV2(BaseModel):
    name: str
    count: int = 0
    confidence: float = 1.0
    children: list["File | FolderV2"] = []

    @property
    def children_map(self):
        return {child.name: child for child in self.children}


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
