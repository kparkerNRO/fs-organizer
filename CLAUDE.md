# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FS-Organizer is a file system organization tool that analyzes messy directory structures (particularly from Patreon creators) and reorganizes them into consistent hierarchies. The system processes folders through a multi-stage pipeline: gather → postprocess → classify → group → generate folder hierarchy.

## Architecture

The project consists of two main components:

### Backend (organizer/)
- **CLI Pipeline**: `organizer.py` provides typer commands for each stage
- **Data Models**: SQLAlchemy models in `data_models/` for database operations
- **Processing Pipeline**: `pipeline/` contains gather, classify, categorize modules
- **Grouping Logic**: `grouping/` handles folder similarity detection and merging
- **FastAPI Server**: `organizer_api.py` serves data to frontend

### Frontend (frontend/)
- **React + TypeScript**: Vite-based SPA with styled-components
- **Two Main Views**: Categories page (folder grouping) and Folder Structure page
- **Mock Mode**: Built-in mock data system for development without backend

## Build & Development Commands

### Backend (organizer/)
```bash
# Install dependencies
pip install -e .

# Run pipeline stages
python organizer.py gather <input_path> <output_path>
python organizer.py postprocess <db_path>
python organizer.py classify <db_path>
python organizer.py group <db_path>
python organizer.py folders <db_path>

# Start API server
fastapi dev organizer_api.py

# Testing
pytest
pytest <file>::<test_name> -v
pytest -k "pattern"

# Code quality
ruff check .
ruff format .
```

### Frontend (frontend/)
```bash
# Development
npm run dev          # Start dev server on http://localhost:5173
npm run build        # Build for production
npm run lint         # ESLint
npm run preview      # Preview production build
```

## Key Data Flow

1. **Gather**: Scans filesystem, stores folder/file metadata in SQLite database
2. **Postprocess**: Cleans filenames and adds derived metadata
3. **Classify**: Labels folders as variant/collection/subject/uncertain based on heuristics
4. **Group**: Uses NLP similarity to group near-duplicate folder names
5. **Categorize**: Generates new folder hierarchy based on classifications and groups

## Database Schema

- **GroupCategory**: Contains grouped folder categories with confidence scores
- **GroupCategoryEntry**: Individual folders assigned to groups
- Uses SQLite with timestamped runs in `outputs/` directory
- Latest run symlinked as `outputs/latest/latest.db`

## API Endpoints

- `GET /groups`: Paginated category data with sorting/filtering
- Frontend hardcoded to mock mode (`isMockMode = true` in api.ts)

## Development Notes

- Backend uses strict typing with pydantic models
- Frontend components are functional with React hooks
- Mock data system allows frontend development without backend
- Database operations use SQLAlchemy ORM
- Testing uses pytest with parametrized tests
- Code formatting enforced by ruff (Python) and ESLint (TypeScript)