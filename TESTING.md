# Testing Guide

This document explains how to run tests in the fs-organizer project with and without large ML dependencies.

## Overview

The test suite is organized to allow running tests without downloading large ML dependencies (sentence-transformers, setfit, datasets). This is useful for:
- Faster CI/CD pipelines
- Development environments with limited resources
- Quick validation of non-ML functionality

## Dependency Groups

The project uses uv's dependency groups to manage optional ML dependencies:

- **Main dependencies**: Core dependencies required for basic functionality (includes scikit-learn)
- **classifier group**: Large ML dependencies (sentence-transformers, setfit, datasets)
- **dev group**: Development tools (debugpy, factory-boy, pre-commit, pyright, ty)

## Installing Dependencies

### Without ML dependencies (lightweight)
```bash
cd organizer
uv sync
```

### With ML dependencies (full suite)
```bash
cd organizer
uv sync --group classifier
```

## Running Tests

### Run only non-ML tests (fast, no large dependencies required)
```bash
cd organizer
# Ignore test files that import ML-dependent modules
pytest --ignore=fine_tuning/services/predict_test.py \
       --ignore=fine_tuning/services/train_test.py \
       --ignore=stages/grouping/tag_decomposition_test.py \
       -m "not ml"
```

### Run only ML tests (requires classifier dependencies)
```bash
cd organizer
uv sync --group classifier
pytest -m "ml"
```

### Run all tests
```bash
cd organizer
uv sync --group classifier
pytest
```

## Test Markers

Tests are marked with pytest markers to indicate their dependency requirements:

- `@pytest.mark.ml`: Tests that require large ML dependencies (sentence-transformers, setfit, datasets)

### Files with ML-dependent tests:
- `stages/grouping/tag_decomposition_test.py` - Tests for semantic tag decomposition (uses sentence-transformers)
- `fine_tuning/services/train_test.py` - Tests for model training (uses setfit, datasets)
- `fine_tuning/services/predict_test.py` - Tests for model prediction (uses setfit)
- `fine_tuning/services/evaluation_test.py` - Tests for model evaluation metrics

**Note**: Some test files import modules that have ML dependencies at the module level. These test files cannot be collected without installing the classifier dependencies, even though they are marked with `@pytest.mark.ml`. This is a known limitation - to fully separate test collection, the source modules would need lazy/conditional imports.

## Example Workflow

### Development without ML features
```bash
# Install lightweight dependencies
cd organizer
uv sync

# Run tests (excluding ML tests)
pytest --ignore=fine_tuning/services/predict_test.py \
       --ignore=fine_tuning/services/train_test.py \
       --ignore=stages/grouping/tag_decomposition_test.py \
       -m "not ml"
```

### Full ML development
```bash
# Install all dependencies including ML
cd organizer
uv sync --group classifier

# Run all tests
pytest

# Or run only ML tests
pytest -m "ml"
```

## CI/CD Considerations

For CI/CD pipelines, you can configure separate jobs:

1. **Fast job** (runs on every commit):
   ```bash
   uv sync
   pytest --ignore=fine_tuning/services/predict_test.py \
          --ignore=fine_tuning/services/train_test.py \
          --ignore=stages/grouping/tag_decomposition_test.py \
          -m "not ml"
   ```

2. **Full job** (runs nightly or on main branch):
   ```bash
   uv sync --group classifier
   pytest
   ```

## Additional pytest options

```bash
# Verbose output
pytest -v

# Show print statements
pytest -s

# Run specific test file
pytest stages/grouping/tag_decomposition_test.py

# Run specific test
pytest stages/grouping/tag_decomposition_test.py::test_full_decomposition_pipeline
```
