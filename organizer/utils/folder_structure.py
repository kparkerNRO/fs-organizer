from typing import Any
from collections.abc import Sequence
from api.api import FolderV2, File


def insert_file_in_structure(
    folder_structure: FolderV2,
    file: Any,  # Any object with id, file_name, and file_path attributes
    parts: Sequence[str | tuple[str, float]],
    new_path: str | None = None,
):
    current_representation = folder_structure

    for component in parts:
        if isinstance(component, tuple):
            component, confidence = component
        else:
            confidence = 1
        if component not in current_representation.children_map.keys():
            current_representation.children.append(
                FolderV2(name=component, confidence=confidence)
            )
        current_representation = current_representation.children_map[component]

    file_id = getattr(file, "node_id", None)
    if file_id is None:
        file_id = getattr(file, "id", None)

    file_name = getattr(file, "name", None)
    if file_name is None:
        file_name = getattr(file, "file_name", None)

    file_path = getattr(file, "abs_path", None)
    if file_path is None:
        file_path = getattr(file, "file_path", None)

    current_representation.children.append(
        File(
            id=file_id,  # type: ignore[arg-type]  # ty bug: SQLAlchemy ORM attribute should be int
            name=file_name,  # type: ignore[arg-type]  # ty bug: SQLAlchemy ORM attribute should be str
            originalPath=file_path,  # type: ignore[arg-type]  # ty bug: SQLAlchemy ORM attribute should be str
            newPath=new_path,
        )
    )
