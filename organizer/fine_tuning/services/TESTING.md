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
# Fast tests (no ML dependencies needed)
pytest fine_tuning/services/ -m "not ml"

# Slower tests (requires ML dependencies)
pytest fine_tuning/services/ -m "ml"
```

This allows you to get quick feedback on most tests while deferring the slower ML-dependent tests to a separate job or later stage.
