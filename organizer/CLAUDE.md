# FS-Organizer Project Helper

## Build/Test Commands
- Install dependencies: `pip install -e .`
- Run tests: `pytest`
- Run single test: `pytest grouping/group_test.py::test_common_token_grouping -v`
- Run tests with pattern: `pytest -k "test_name_pattern"`
- Lint code: `ruff check .`
- Format code: `ruff format .`

## Code Style Guidelines
- **Imports**: Group standard library, third-party, and local imports with blank lines
- **Type Annotations**: Use typing module; annotate function parameters and return types
- **Docstrings**: Use triple quotes for docstrings; explain function purpose and parameters
- **Naming**: snake_case for functions/variables; CamelCase for classes
- **Error Handling**: Use explicit exception handling with specific exception types
- **Testing**: Use pytest with parametrized tests for multiple test cases
- **Line Length**: Maximum 100 characters
- **Function Length**: Keep functions short and focused on a single task
- **Formatting**: Use ruff for consistent code formatting