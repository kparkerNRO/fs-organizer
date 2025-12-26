"""
The workhorse of the classification system - given a file set,
compute features and finetune the model

"""
from pathlib import Path



def fine_tune_model(db_path: Path):
    index_path = db_path / "index.db"
    work_path = db_path / "classify.db"

    # use SQLAlchemy to interact with the sqlite databases. Use session context manager

    # create the new classification run


    # run feature extraction


    # finish the session

# TODO: add a TYPER route here to update tunings