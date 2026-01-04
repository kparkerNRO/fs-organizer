from collections.abc import Sequence

from api.api import File, FolderV2
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


def sort_folder_structure(folder_data: dict) -> dict:
    """
    Recursively sort folder structure by folder/file names
    """
    if not isinstance(folder_data, dict):
        return folder_data

    # Create a new FolderV2 object to ensure proper structure
    if "name" in folder_data and "children" in folder_data:
        # This is a folder object
        sorted_children = []

        # Sort children by name
        children = folder_data.get("children", [])
        if children:
            # Separate files and folders
            files = [child for child in children if "id" in child]
            folders = [child for child in children if "id" not in child]

            # Sort files by name
            files.sort(key=lambda x: x.get("name", "").lower())

            # Sort folders by name and recursively sort their children
            folders.sort(key=lambda x: x.get("name", "").lower())
            for folder in folders:
                sorted_children.append(sort_folder_structure(folder))

            # Add files after folders
            sorted_children.extend(files)

        # Return sorted folder
        return {**folder_data, "children": sorted_children}

    return folder_data
