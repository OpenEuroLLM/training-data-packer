# Contributing

Everyone is welcome to contribute, both in code and discussion.

## Setup

```bash
uv sync --extra dev
uv run pre-commit install
```

## Making changes

- Keep PRs small and focused on a single change.
- Write tests for new features and bug fixes — pre-commit hooks will enforce tests with pytest.
- Follow existing code style — pre-commit hooks will enforce formatting with ruff.

## Submitting a PR

1. Open an issue first for non-trivial changes.
2. Branch off `main`.
