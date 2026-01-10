# Tools
This project uses the following tools:
* `uv` for depdendency management
* `just` for task running
* `ruff` for formatting
* `ty` for type checking
* `typer` for CLI tools
* `pydantic-settings` for loading system settings in from the environment 
* `fastapi` for API serving

# Conventions
* Use context managers for sqlalchemy sessions
* Use `type | None` instead of `Optional[type]`, and prefer python primitives rather than importing from typing
* always run tests and `just pre-commit` before declaring a refactoring task done
* Use enums instead of string literals
* always put imports at the top of files
* Prefer catching explicit exceptions to general ones. Do not implement this pattern except at the top level entrypoint to catch exceptions before crashing the system:
```
try:
    ....
except Exception as e:
    ...
```
* `just ci-check` **MUST** pass before the code is merged to main
* System design and review documentation should be put in 

## Testing conventions
* Use pytest with fixtures and parameterized tests for test cases
* Use factories and the minimum possible changes defined in test data, rather than defining all fields
* Use dependency injection with mocked or subclassed classes rather than patching. Refactor functions to use dependency injection rather than mocking
* prefer exact matches (assert value == var) over containment checks (assert value in var)
* prefer exact matches (assert value == var) over count checks (assert value > var, assert value < var)
* Do not test core python functionality - avoid tests that look like:
```
    def test_node_creation(self):
        """Test creating a node HierarchyItem."""
        item = HierarchyItem(
            id="node-123",
            name="Documents",
            type="node",
            originalPath="/home/user/Documents",
        )
        assert item.id == "node-123"
        assert item.name == "Documents"
        assert item.type == "node"
        assert item.originalPath == "/home/user/Documents"
```

# Other notes
* Several tests depend on heavy-weight ML dependencies. In resource-constraind environments (eg: claude code web), you may run only the tests specific to your change to avoid downloading those tests.