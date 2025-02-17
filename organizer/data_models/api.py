from enum import Enum
from typing import List, Optional
from pydantic import BaseModel



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