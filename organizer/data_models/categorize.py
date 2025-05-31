from dataclasses import dataclass


@dataclass
class Category:
    id: int
    name: str
    classification: str


# @dataclass
# class Group():
#     id: int
#     name: str
#     item_count: int
#     categories: dict[str, Category]
#     folders: list[Folder]
