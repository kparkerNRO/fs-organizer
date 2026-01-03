from collections.abc import Sequence
from api.api import FolderV2, File
from storage.index_models import Node


def insert_file_in_structure(
    folder_structure: FolderV2,
    file: Node,
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

    current_representation.children.append(
        File(
            id=file.node_id,
            name=file.name,
            originalPath=file.abs_path,
            newPath=new_path,
        )
    )
