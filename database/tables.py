from sqlalchemy import Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.sqlite.json import JSON

Base = declarative_base()

class Runs(Base):
    __tablename__ = 'runs'

    id = Column(Integer, primary_key=True)
    root_folder = Column(String(255), nullable=False)

class Exports(Base):
    __tablename__ = 'exports'

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('runs.id'))
    export_stage = Column(String(255), nullable=False)
    folder_structure = Column(JSON, nullable=True)

class Tags(Base):
    __tablename__ = 'tags'

    id = Column(Integer, primary_key=True)
    original_name = Column(String(255), nullable=False)
    normalized_name = Column(String(255), nullable=True)
    final_name = Column(String(255), nullable=True)

class Files(Base):
    __tablename__ = 'files'

    id = Column(Integer, primary_key=True)
    export_id = Column(Integer, ForeignKey('exports.id'))
    file_name = Column(String(255), nullable=False)
    zip_parent = Column(String(255), nullable=True)
    original_path = Column(String(255), nullable=False)
    current_path = Column(String(255), nullable=True)
    tags = Column(JSON, nullable=True)