from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


class File(BaseModel):
    id: int
    name: str
    confidence: int = 100
    possibleClassifications: list[str] = []
    originalPath: str
    newPath: str

class FolderV2(BaseModel):
    name: str
    count: int = 0
    confidence: int = 100
    children: list[File] = []

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
    name = "name"
    count = "count"
    confidence = "confidence",
    id = "id"

class SortOrder(str, Enum):
    asc = "asc"
    desc = "desc"