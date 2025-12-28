from api.api import FolderV2, File

from data_models.database import File as dbFile


def insert_file_in_structure(
    folder_structure: FolderV2,
    file: dbFile,
    parts: list[str | tuple] | tuple[str, ...],
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
            id=file.id,  # type: ignore[arg-type]  # ty bug: SQLAlchemy ORM attribute should be int
            name=file.file_name,  # type: ignore[arg-type]  # ty bug: SQLAlchemy ORM attribute should be str
            originalPath=file.file_path,  # type: ignore[arg-type]  # ty bug: SQLAlchemy ORM attribute should be str
            newPath=new_path,
        )
    )
