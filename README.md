# FS-Organizer

> A work-in-progress tool for taming the chaos of Patreon RPG resource collections (and my personal AI experimentation and learn React playground)

## The Problem

If you've ever subscribed to multiple TTRPG creators on Patreon, you know the pain: hundreds of folders with inconsistent naming, duplicates, variants of the same content, and a general lack of structure that makes finding anything a nightmare. You end up with things like:

- `Fantasy_Tavern_01`
- `fantasy tavern 1`
- `Fantasy Tavern (Night Version)`
- `FantasyTavern_Day_02`

All scattered across your filesystem with no rhyme or reason.

This project is my attempt to build something that can automatically analyze these messy collections and reorganize them into something actually usable. Also, it's a great excuse to experiment with different AI technologies and see what works (and what doesn't).

## What It Does

FS-Organizer uses a multi-stage pipeline to process your chaotic folder structures:

1. **Gather** - Scans your filesystem and catalogs everything into a SQLite database
2. **Postprocess** - Cleans up filenames and extracts useful metadata
3. **Classify** - Figures out what kind of folder each one is (collection, variant, subject, etc.)
4. **Group** - Uses NLP similarity detection to find near-duplicates and related folders
5. **Generate Hierarchy** - Proposes a new, sensible folder structure

The system comes with both a CLI for running the pipeline and a web UI for reviewing and managing the results.

## Project Structure

```
organizer/          # Python backend
  ├── organizer.py  # Main CLI pipeline
  ├── organizer_api.py  # FastAPI server
  ├── pipeline/     # Processing stages
  ├── grouping/     # Similarity detection
  └── data_models/  # SQLAlchemy models

frontend/           # React + TypeScript UI
  └── src/
      └── components/  # Browse and manage grouped folders
```

## Quick Start

### Backend

```bash
cd organizer
pip install -e .

# Run the full pipeline
python organizer.py gather /path/to/messy/folders outputs/my-run
python organizer.py postprocess outputs/my-run/my-run.db
python organizer.py classify outputs/my-run/my-run.db
python organizer.py group outputs/my-run/my-run.db
python organizer.py folders outputs/my-run/my-run.db

# Or start the API server to use with the frontend
fastapi dev organizer_api.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI runs at http://localhost:5173 and has a built-in mock mode so you can explore it without running the backend.

## Current Status

This is very much a work in progress. At any given point, saying that it "works" is probably a strech:

- The classification heuristics could be smarter
- The similarity detection sometimes groups things that shouldn't be grouped
- The generated hierarchy is functional but not always intuitive
- The UI is basic but gets the job done

I'm actively experimenting with different approaches, so expect things to change frequently.

## Tech Stack

**Backend:**
- Python with typer for CLI
- SQLAlchemy for database operations
- FastAPI for the API server
- NLTK for text similarity
- pytest for testing
- ruff for code formatting

**Frontend:**
- React + TypeScript
- Vite for building
- styled-components for styling
- Mock data system for development

## Why This Exists

Honestly? I got tired of manually organizing hundreds of gigabytes of TTRPG maps and assets. And I wanted an excuse to play around with different AI and NLP techniques in a real-world problem that actually affects me.

If this ends up being useful to other people dealing with similar organizational nightmares, that's a bonus.

## Contributing

This is a personal project and pretty rough around the edges, but if you have ideas or want to experiment with it, feel free to open an issue or PR.

## License

MIT - do whatever you want with it.
