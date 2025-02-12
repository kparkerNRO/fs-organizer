from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import shutil
from typing import Optional
from organizer.data_models.database import setup_gather
from pipeline.gather import gather_folder_structure_and_store, clean_file_name_post
from pipeline.classify import classify_folders
from grouping.group import categorize

app = FastAPI()

class GatherRequest(BaseModel):
    base_path: str
    output_dir: str

@app.post("/gather")
async def gather_endpoint(request: GatherRequest):
    """
    Gather folder structure data and store in database
    """
    try:
        base_path = Path(request.base_path)
        output_dir = Path(request.output_dir)
        
        if not base_path.exists() or not base_path.is_dir():
            raise HTTPException(status_code=400, detail="Invalid base path")
            
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped run directory
        timestamp_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = output_dir / timestamp_str
        run_dir.mkdir()

        # Set up and populate database
        db_path = run_dir / "run_data.db"
        setup_gather(db_path)
        gather_folder_structure_and_store(base_path, db_path)

        # Handle latest symlink
        latest_dir = output_dir / "latest"
        latest_db = latest_dir / "latest.db"
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        latest_dir.mkdir()
        shutil.copy2(db_path, latest_db)

        return {
            "status": "success",
            "run_dir": str(run_dir),
            "db_path": str(db_path),
            "latest_db": str(latest_db)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("groups")
async def get_groups():
    """
    Get the pre-calculated grouping
    """
    