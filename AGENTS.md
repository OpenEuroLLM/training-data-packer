# AGENTS.md

## Setup

```bash
uv sync --dev
uv run pre-commit install
```

## Run
There are four components in the packer:
- `oellm-package-data` - collect and package data files
- `oellm-collect-metrics` - collect metrics from packaged data
- `oellm-propella-structure` - create propella structure from data files
- `oellm-propella-merge` - merge propella structure files

To run the packer:
```bash
uv run oellm-package-data
```

To collect metrics:
```bash
uv run oellm-collect-metrics
```

To create propella structure:
```bash
uv run oellm-propella-structure
```

To merge propella structure:
```bash
uv run oellm-propella-merge
```

## Development

- Python 3.14+ required (enforced by `pyproject.toml`)
- Uses `uv` for dependency management
- Format the code with `uv run ruff format`
- Check coding style with `uv run ruff check`
- Run tests with `uv run --with pytest pytest`

## Code style guidelines

- Follow pep-8
- Prefer dubble quote(")
- Use space as indent style
- Have a new line end of file
- Use magic trailing commas
- Prefix private functions with underscore(_)

## Documentation

- Write doc strings for functions, classes and modules.
- The first line of the docstring should be in an imperative mood.
- Write doc strings in reST style, for example:

```python
"""This is a reST style.

:param param1: this is a first param
:param param2: this is a second param
:returns: this is a description of what is returned
:raises keyError: raises an exception

"""
```
