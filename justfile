# FS-Organizer justfile
# Run `just` to see available commands

# Default recipe: list all available commands
default:
    @just --list

# ============================================
# Setup & Initialization
# ============================================

# Initialize project: setup Node (via nvm), uv, and install all dependencies
init:
    @echo "=== Initializing FS-Organizer Development Environment ==="
    @echo ""
    @echo "1. Checking Node.js setup..."
    @if command -v nvm >/dev/null 2>&1; then \
        echo "✓ nvm found"; \
    else \
        echo "✗ nvm not found. Please install nvm first:"; \
        echo "  https://github.com/nvm-sh/nvm#installing-and-updating"; \
        exit 1; \
    fi
    @echo "  Switching to Node 20 (from .nvmrc)..."
    @bash -c "source ~/.nvm/nvm.sh && cd frontend && nvm install && nvm use"
    @echo ""
    @echo "2. Checking uv setup..."
    @if command -v uv >/dev/null 2>&1; then \
        echo "✓ uv found (version: $$(uv --version))"; \
    else \
        echo "✗ uv not found. Installing uv..."; \
        curl -LsSf https://astral.sh/uv/install.sh | sh; \
    fi
    @echo ""
    @echo "3. Installing backend dependencies..."
    just install-backend
    @echo ""
    @echo "4. Installing frontend dependencies..."
    just install-frontend
    @echo ""
    @echo "=== ✓ Initialization complete! ==="
    @echo ""
    @echo "Next steps:"
    @echo "  - Run 'just dev' to start the Electron app"
    @echo "  - Run 'just dev-all' to start both backend API and Electron app"
    @echo "  - Run 'just --list' to see all available commands"

# ============================================
# Backend (organizer/)
# ============================================

# Install backend dependencies
install-backend:
    cd organizer && uv sync

# Run all pipeline stages on input directory
pipeline input output:
    cd organizer && uv run python organizer.py gather {{input}} {{output}}
    cd organizer && uv run python organizer.py postprocess {{output}}/latest/latest.db
    cd organizer && uv run python organizer.py classify {{output}}/latest/latest.db
    cd organizer && uv run python organizer.py group {{output}}/latest/latest.db
    cd organizer && uv run python organizer.py folders {{output}}/latest/latest.db

# Gather: scan filesystem and store metadata
gather input output:
    cd organizer && uv run python organizer.py gather {{input}} {{output}}

# Postprocess: clean filenames and add metadata
postprocess db_path:
    cd organizer && uv run python organizer.py postprocess {{db_path}}

# Classify: label folders as variant/collection/subject/uncertain
classify db_path:
    cd organizer && uv run python organizer.py classify {{db_path}}

# Group: use NLP similarity to group near-duplicate folder names
group db_path:
    cd organizer && uv run python organizer.py group {{db_path}}

# Folders: generate new folder hierarchy
folders db_path:
    cd organizer && uv run python organizer.py folders {{db_path}}

# Start FastAPI development server
api:
    cd organizer && uv run fastapi dev organizer_api.py

# Run all backend tests
test-backend:
    cd organizer && uv run pytest

# Run specific backend test
test-backend-file file:
    cd organizer && uv run pytest {{file}} -v

# Run backend tests matching pattern
test-backend-pattern pattern:
    cd organizer && uv run pytest -k "{{pattern}}"

# Check backend code quality with ruff
lint-backend:
    cd organizer && uv run ruff check .

# Format backend code with ruff
format-backend:
    cd organizer && uv run ruff format .

# run the type checker on the backend code
type-backend:
    cd organizer && uv run ty check .

# Lint and format backend code
fix-backend: lint-backend format-backend

# Run pre-commit checks (same as pre-commit hooks)
pre-commit:
    cd organizer && uv run ruff check --fix .
    cd organizer && uv run ruff format .

# ============================================
# Frontend - Electron App (frontend/)
# ============================================

# Install frontend dependencies
install-frontend:
    cd frontend && npm install

# Start Electron app in development mode
dev:
    cd frontend && npm run electron:dev

# Start web-only development server (without Electron)
dev-web:
    cd frontend && npm run dev

# Build frontend for Electron
build-frontend:
    cd frontend && npm run build:electron

# Build Electron distributables (packaged app)
build-electron-dist:
    cd frontend && npm run electron:dist

# Package Electron app
build-electron-pack:
    cd frontend && npm run electron:pack

# Lint frontend code
lint-frontend:
    cd frontend && npm run lint

# Run frontend tests
test-frontend:
    cd frontend && npm run test:run

# Run frontend tests with UI
test-frontend-ui:
    cd frontend && npm run test:ui

# Run frontend tests with coverage
test-frontend-coverage:
    cd frontend && npm run test:coverage

# ============================================
# Combined Commands
# ============================================

# Install all dependencies (backend + frontend)
install: install-backend install-frontend

# Lint and format all code
lint: lint-backend lint-frontend

# Format all code
format: format-backend

# Run all tests
test: test-backend test-frontend

# Start both API server and Electron app
dev-all:
    @echo "Starting backend API and Electron app..."
    @echo "Backend API will be on http://localhost:8000"
    @echo "Electron app will launch automatically"
    just api & just dev

# Build everything for production (backend + Electron app)
build: build-frontend

# Build and package Electron app for distribution
build-dist: build-electron-dist

# ============================================
# Utilities
# ============================================

# Clean build artifacts
clean:
    rm -rf frontend/dist
    rm -rf frontend/dist-electron
    rm -rf frontend/release
    rm -rf frontend/node_modules/.vite
    find organizer -type d -name __pycache__ -exec rm -rf {} +
    find organizer -type d -name "*.egg-info" -exec rm -rf {} +
