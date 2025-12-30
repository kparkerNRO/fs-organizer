# Testing Guide

## Test Organization

Tests in this directory are organized with pytest markers to separate tests based on their dependencies.

### Markers

- `@pytest.mark.ml`: Tests that require large ML dependencies (sentence-transformers, setfit, datasets)

### Running Tests

Run all tests (requires ML dependencies):
```bash
pytest fine_tuning/services/
```

Run only tests that **don't** require ML dependencies:
```bash
# Option 1: Ignore ML test files (faster, avoids import errors)
pytest --ignore=fine_tuning/services/train_test.py --ignore=fine_tuning/services/predict_test.py

# Option 2: Use markers (requires ML deps to be importable)
pytest fine_tuning/services/ -m "not ml"
```

Run only tests that **do** require ML dependencies:
```bash
pytest fine_tuning/services/ -m "ml"
```

### ML-Dependent Tests

The following test files require ML dependencies:
- `train_test.py` - Tests for model training functions (requires setfit, datasets)
- `predict_test.py` - Tests for prediction functions (requires setfit, sentence-transformers via imports)

The following test files do **not** require ML dependencies:
- `common_test.py` - Common utility functions
- `sampling_test.py` - Training sample selection
- `evaluation_test.py` - Prediction evaluation metrics
- `feature_extraction_test.py` - Feature extraction

### CI/CD Integration

For faster CI pipelines, you can run non-ML tests first:
```bash
# Fast tests (no ML dependencies needed) - ignore ML test files to avoid imports
pytest --ignore=fine_tuning/services/train_test.py --ignore=fine_tuning/services/predict_test.py

# Slower tests (requires ML dependencies)
pytest -m "ml"
```

**Note**: Using `--ignore` is recommended over `-m "not ml"` for the non-ML tests because pytest imports all test files during collection, even if they're marked to skip. The `--ignore` flag prevents pytest from importing ML test files entirely, avoiding import errors when ML dependencies aren't installed.

This allows you to get quick feedback on most tests while deferring the slower ML-dependent tests to a separate job or later stage.
