# Tools and conventions
* Use typer for CLI tools
* Use context managers for sqlalchemy sessions
* Use `type | None` instead of `Optional[type]`, and prefer python primitives rather than importing from typing
* always run tests and `just pre-commit` before declaring a refactoring task done

## Testing conventions
* Use pytest with fixtures and parameterized tests for test cases
* Use factories and the minimum possible changes defined in test data, rather than defining all fields
* Use dependency injection with mocked or subclassed classes rather than patching. Refactor functions to use dependency injection rather than mocking
* prefer exact matches (assert value == var) over containment checks (assert value in var)
* prefer exact matches (assert value == var) over count checks (assert value > var, assert value < var)

# Running the application
* Inside organizer, use `uv run` instead of `python` and `pytest`