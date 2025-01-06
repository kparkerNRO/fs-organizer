
from dataclasses import dataclass

@dataclass
class Category():
    id: int
    name: str
    classification: str

@dataclass
class Folder():
    id: int
    name: str
    parent_path: str
    path: str
    classification: str

    categories: dict[str, Category]
    original_name: str

@dataclass
class CategorySummary():
    id: int
    name: str
    total_count: int
    classification_counts: dict[str, int]