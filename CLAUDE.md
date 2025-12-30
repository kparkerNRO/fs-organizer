# Tools and conventions
* Use typer for CLI tools
* Use context managers for sqlalchemy sessions
* Use pytest with fixtures and parameterized tests for test cases
* Use `type | None` instead of `Optional[type]`, and prefer python primitives rather than importing from typing
* Use dependency injection with mocked or subclassed classes rather than patching
* always run tests and `just pre-commit` before declaring a refactoring task done