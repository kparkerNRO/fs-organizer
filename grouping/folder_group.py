from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import Dict, List
from pathlib import Path

from database import Folder
from grouping.helpers import common_token_grouping


def get_folder_groups(session: Session) -> Dict[str, List[Folder]]:
    """
    Retrieve folders grouped by parent path where there is more than one folder 
    in the group.
    
    Returns:
        Dict[str, List[Folder]]: Dictionary with parent paths as keys and lists of folders as values
    """
    # First find parent paths that have multiple folders
    parent_counts = (
        session.query(
            Folder.parent_path,
            func.count(Folder.id).label('folder_count')
        )
        .group_by(Folder.parent_path)
        .having(func.count(Folder.id) > 1)
        .subquery()
    )
    
    # Then get all folders that belong to these parent paths
    folders = (
        session.query(Folder)
        .join(
            parent_counts,
            Folder.parent_path == parent_counts.c.parent_path
        )
        .order_by(Folder.parent_path, Folder.folder_name)
        .all()
    )
    
    # Group the folders by parent path
    grouped_folders = {}
    for folder in folders:
        grouped_folders.setdefault(folder.parent_path, []).append(folder)
        
    return grouped_folders

def process_folders(session: Session):
    """
    Process groups of folders and update the classification in the database.
    
    Args:
        session (Session): SQLAlchemy session
    """

    grouped_folders = get_folder_groups(session)

    for parent_path, folders in grouped_folders.items():
        folder_names = [folder.folder_name for folder in folders]
        folder_name_to_folder = {folder.folder_name: folder for folder in folders}  
        token_grouping = common_token_grouping(folder_names)
        for name, category in token_grouping.items():
            for folder in folders:
                if folder.folder_name == name:
                    folder.classification = category
            
    session.commit()