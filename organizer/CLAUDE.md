# Backend Context (Python + SQLAlchemy)

Context for working with the organizer backend.

## Module Structure

```
organizer/
├── organizer.py          # CLI entry point (typer commands)
├── organizer_api.py      # FastAPI server
├── data_models/
│   ├── database.py       # SQLAlchemy session management
│   ├── folder.py         # Folder table model
│   ├── gather.py         # FileMetadata model
│   ├── classify.py       # FolderClassification model
│   └── categorize.py     # GroupCategory, GroupCategoryEntry models
├── pipeline/
│   ├── gather.py         # Filesystem scanning (Stage 1)
│   ├── classify.py       # Folder classification (Stage 3)
│   ├── categorize.py     # Hierarchy generation (Stage 5)
│   └── folder_reconstruction.py  # Folder hierarchy logic
├── grouping/
│   ├── group.py          # Main grouping orchestration
│   ├── nlp_grouping.py   # NLTK-based similarity detection
│   ├── tag_decomposition.py  # Token extraction and analysis
│   ├── group_cleanup.py  # Post-processing group refinement
│   └── helpers.py        # Shared utilities
└── utils/
    ├── filename_utils.py # Name cleaning and parsing
    └── common.py         # Shared utilities
```

## Pipeline Stages

1. **gather** (`pipeline/gather.py`) → Scan filesystem → Create FileMetadata records
2. **postprocess** (in classify) → Clean names → Add metadata
3. **classify** (`pipeline/classify.py`) → Label folders → variant/collection/subject/uncertain
4. **group** (`grouping/group.py`) → NLP similarity → GroupCategory + GroupCategoryEntry
5. **folders** (`pipeline/categorize.py`) → Generate hierarchy → New folder structure

## Database Models (SQLAlchemy)

### Core Tables
- **Folder**: Basic folder info (path, name, parent)
- **FileMetadata**: File scan results (name, size, type)
- **FolderClassification**: Classification labels (variant, collection, etc.)
- **GroupCategory**: Grouped folder categories (name, confidence_score)
- **GroupCategoryEntry**: Folder assignments to groups (folder_id, category_id)

### Database Location
- `outputs/<timestamp>/<name>.db` (SQLite)
- Latest symlinked at `outputs/latest/latest.db`
- Session management in `data_models/database.py`

## Key Patterns

### CLI Commands (typer)
```python
# organizer.py
@app.command()
def gather(input_path: str, output_path: str):
    """Scan filesystem and create database."""
    # Implementation in pipeline/gather.py
```

### Database Sessions
```python
from data_models.database import get_session

with get_session(db_path) as session:
    folders = session.query(Folder).all()
```

### Testing
- **Framework**: pytest
- **Pattern**: Parametrized tests for multiple cases
- **Example**: See `grouping/group_test.py`, `pipeline/gather_test.py`
- **Naming**: `test_*.py` or `*_test.py`

### Grouping Algorithm
1. **Extract tokens** (`tag_decomposition.py`) → Break folder names into meaningful parts
2. **Calculate similarity** (`nlp_grouping.py`) → NLTK-based matching
3. **Create groups** (`group.py`) → Cluster similar folders
4. **Clean up** (`group_cleanup.py`) → Refine and validate groups

## Code Conventions

### Python Style
- **Commands**: Always use `uv run` prefix (e.g., `uv run pytest`)
- **Imports**: Standard lib → Third-party → Local (with blank lines)
- **Typing**: Full type annotations (parameters + return types)
- **Naming**: snake_case functions/variables, PascalCase classes
- **Line length**: 100 characters max
- **Formatting**: ruff (auto-format on save recommended)

### Type Annotations
```python
from typing import List, Optional
from sqlalchemy.orm import Session

def get_folders(session: Session, parent_id: Optional[int] = None) -> List[Folder]:
    """Get folders with optional parent filter."""
    return session.query(Folder).filter_by(parent_id=parent_id).all()
```

### Testing Pattern
```python
import pytest

@pytest.mark.parametrize("input_name,expected_tokens", [
    ("Fantasy_Tavern_01", ["fantasy", "tavern", "01"]),
    ("fantasy tavern 1", ["fantasy", "tavern", "1"]),
])
def test_token_extraction(input_name: str, expected_tokens: list[str]):
    result = extract_tokens(input_name)
    assert result == expected_tokens
```

## Common Tasks

### Add New Pipeline Stage
1. Create module in `pipeline/new_stage.py`
2. Define function with session parameter
3. Add command in `organizer.py` using typer
4. Add tests in `pipeline/new_stage_test.py`

### Modify Database Schema
1. Update model in `data_models/`
2. Models auto-create tables (no migrations currently)
3. Delete old database or add migration logic

### Run Specific Tests
```bash
uv run pytest grouping/group_test.py::test_name -v
uv run pytest -k "similarity" -v
uv run pytest --lf  # Last failed
```

### Debug SQL Queries
```python
from data_models.database import get_session

with get_session(db_path) as session:
    session.execute("SELECT * FROM folders").fetchall()
```

## Critical Files

- `organizer.py` - CLI entry point, all commands defined here
- `data_models/categorize.py` - GroupCategory model (main output)
- `grouping/nlp_grouping.py` - Core similarity algorithm
- `pipeline/classify.py` - Folder classification logic
- `utils/filename_utils.py` - Name cleaning utilities