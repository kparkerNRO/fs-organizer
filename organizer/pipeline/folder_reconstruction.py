import json
from typing import Optional
import os
from sqlalchemy import create_engine, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
import typer
from data_models.database import Folder, GroupCategoryEntry

Base = declarative_base()


def get_parent_folder(
    session: Session, parent_path: str, zip_content=False
) -> Optional[Folder]:
    """Find the parent folder entry based on its path."""

    parent = session.query(Folder).filter(Folder.folder_path == parent_path).first()

    if not parent and zip_content:
        parent_path = parent_path.rsplit("/", 1)[0]
        # If the parent folder is not found, try to find it in the zip content
        parent = session.query(Folder).filter(Folder.folder_path == parent_path).first()

    return parent


def get_category_name(session: Session, folder_id: int, iteration_id) -> str:
    """Get the category name for a folder."""
    categories = (
        session.query(GroupCategoryEntry)
        .distinct(GroupCategoryEntry.processed_name)
        .filter(GroupCategoryEntry.folder_id == folder_id)
        .filter(GroupCategoryEntry.iteration_id == iteration_id)
        .order_by(GroupCategoryEntry.group_id)
        .all()
    )
    names = [category.processed_name for category in categories]
    return "/".join(names)


def build_cleaned_path(session: Session, folder: Folder, target_iteration=0) -> str:
    """
    Recursively build the cleaned path by traversing up the parent hierarchy
    and using cleaned_name at each level.
    """
    cleaned_name = get_category_name(session, folder.id, target_iteration)
    if not folder.parent_path:
        return cleaned_name

    parent = get_parent_folder(
        session, folder.parent_path, zip_content=folder.file_source == "zip_content"
    )
    if not parent:
        return cleaned_name
    if parent.cleaned_path:
        return os.path.join(parent.cleaned_path, cleaned_name)

    parent_cleaned_path = build_cleaned_path(session, parent)
    return os.path.join(parent_cleaned_path, cleaned_name)


def update_folder_paths(db_path: str):
    """Update all folder paths in the database using cleaned names."""
    # Create SQLAlchemy engine
    engine = create_engine(f"sqlite:///{db_path}")

    # Create session
    session = Session(engine)
    max_iterations = session.query(func.max(GroupCategoryEntry.iteration_id)).scalar()

    try:
        # Get all folders
        folders = session.query(Folder).all()
        total_folders = len(folders)

        print(f"Processing {total_folders} folders...")

        # Process each folder
        for i, folder in enumerate(folders, 1):
            if i % 100 == 0:
                print(f"Processed {i}/{total_folders} folders")

            # Skip if no cleaned name
            if not folder.cleaned_name:
                print(f"Warning: Folder {folder.id} has no cleaned_name, skipping")
                continue

            # Build new path using cleaned names
            new_path = build_cleaned_path(
                session, folder, target_iteration=max_iterations
            )
            # print(new_path)

            # Update folder path
            folder.cleaned_path = new_path

            # If this folder is a parent to other folders, we need to update their parent_path
            # child_folders = session.query(Folder).filter(
            #     Folder.parent_path == folder.folder_path
            # ).all()

            # for child in child_folders:
            #     child.parent_path = new_path

        # Commit changes
        session.commit()
        print("Successfully updated all folder paths")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        session.rollback()
        raise

    finally:
        session.close()

def generate_folder_heirarchy_from_cleaned_path(
    session: Session, folder: Folder, target_iteration=0, working_representation={}
) -> dict:
    cleaned_path = folder.cleaned_path
    if not cleaned_path:
        return working_representation
    cleaned_parts = cleaned_path.split("/")

    # Initialize the working representation with the cleaned path
    current_representation = working_representation
    for part in cleaned_parts:
        if part not in current_representation:
            current_representation[part] = {}
        current_representation = current_representation[part]

    return working_representation

def generate_folder_heirarchy(db_path: str):
    print(os.getcwd())
    if not os.path.exists(db_path):
        raise typer.BadParameter(f"Database file not found: {db_path}")

    print(f"Processing database at: {db_path}")
    update_folder_paths(db_path)

    # Generate folder hierarchy from cleaned paths
    engine = create_engine(f"sqlite:///{db_path}")
    session = Session(engine)
    folders = session.query(Folder).all()
    folder_hierarchy = {}
    for folder in folders:
        folder_hierarchy = generate_folder_heirarchy_from_cleaned_path(
            session, folder, working_representation=folder_hierarchy
        )
    session.close()
    print("Folder hierarchy:")
    json_output = json.dumps(folder_hierarchy, indent=4, sort_keys=True)
    print(json_output)

def main(
    db_path: str = typer.Argument(
        "organizer/outputs/latest/latest.db", help="Path to the SQLite database file"
    ),
):
    """
    Update folder paths in the database to use cleaned names.

    Args:
        db_path: Path to the SQLite database file
    """
    # print(os.path.abspath(db_path))
    print(os.getcwd())
    if not os.path.exists(db_path):
        raise typer.BadParameter(f"Database file not found: {db_path}")

    print(f"Processing database at: {db_path}")
    generate_folder_heirarchy(db_path)


if __name__ == "__main__":
    typer.run(main)
