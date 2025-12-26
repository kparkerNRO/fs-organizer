## Project Purpose

Analyzes messy directory structures (particularly Patreon RPG assets) and reorganizes them into consistent hierarchies using a multi-stage pipeline.

## Tools and conventions
* Use typer for CLI tools
* Use context managers for sqlalchemy sessions
* Use pytest with fixtures and parameterized tests for test cases
* Use `type | None` instead of `Optional[type]`, and prefer python primitives rather than importing from typing

## Architecture Overview

```
organizer/               # Python backend
  ├── organizer.py       # CLI entry point (typer commands)
  ├── organizer_api.py   # FastAPI server
  ├── pipeline/          # Processing stages (gather, classify, categorize)
  ├── grouping/          # NLP similarity detection and folder grouping
  └── data_models/       # SQLAlchemy models (GroupCategory, GroupCategoryEntry)

frontend/                # Electron desktop app
  ├── electron/          # Main process (main.ts) and preload (preload.ts)
  └── src/
      ├── components/    # React components (Categories, FolderStructure)
      ├── types/         # TypeScript types
      └── api.ts         # Backend API client (mock mode enabled)
```

## Pipeline Data Flow

1. **gather** → Scan filesystem → SQLite database (`outputs/<timestamp>/<name>.db`)
2. **postprocess** → Clean filenames → Add metadata
3. **classify** → Label folders (variant/collection/subject/uncertain)
4. **group** → NLP similarity → Group near-duplicates
5. **folders** → Generate new hierarchy

## Key Conventions

### Backend (Python + SQLAlchemy)
- **Commands**: Use `uv run` prefix for all Python commands
- **Testing**: `uv run pytest` (parametrized tests, see existing tests for patterns)
- **Database**: SQLite with SQLAlchemy ORM, timestamped outputs in `outputs/`
- **Typing**: Strict typing with pydantic models
- **Code Style**: ruff formatting, snake_case, 100 char line limit

### Frontend (Electron + React + TypeScript)
- **Node Version**: Use `nvm use` (Node 20 LTS via .nvmrc)
- **Commands**: `npm run electron:dev` for development
- **Mock Mode**: `isMockMode = true` in api.ts (no backend needed)
- **Components**: Functional with React hooks, styled-components
- **Testing**: Vitest with @testing-library/react
- **Code Style**: ESLint + Prettier, PascalCase components, camelCase functions

### Electron Architecture
- **main.ts**: App lifecycle, native features, IPC handlers
- **preload.ts**: Secure bridge between main and renderer
- **src/**: React renderer process (Chromium environment)

## Common Tasks

### Run Full Pipeline
```bash
cd organizer
uv run python organizer.py gather <input> outputs/run
uv run python organizer.py postprocess outputs/run/run.db
uv run python organizer.py classify outputs/run/run.db
uv run python organizer.py group outputs/run/run.db
uv run python organizer.py folders outputs/run/run.db
```

### Development
```bash
just init        # First-time setup (nvm + uv + deps)
just dev         # Start Electron app
just dev-all     # Start backend API + Electron app
just test        # Run all tests
```

## Critical Files

- `organizer/data_models/` - Database schema (GroupCategory, GroupCategoryEntry)
- `organizer/pipeline/` - Core processing logic
- `organizer/grouping/` - Similarity detection algorithms
- `frontend/src/api.ts` - API client and mock data
- `frontend/src/components/` - Main UI components
- `justfile` - Task runner commands