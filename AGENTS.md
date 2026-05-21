# AGENTS.md

## Setup

```bash
uv sync
```

## Run
There are two components in the packer:
- `oellm-package-data` - collect and package data files
- `oellm-package-merge` - merges the data files into a larger files

To run the packer:
```bash
uv run oellm-package-data
```

Too run the merger
```bash
uv run oellm-package-merge
```

## Development

- Python 3.14+ required (enforced by `.python-version` and `pyproject.toml`)
- Uses `uv` for dependency management
- Format the code with `uv run ruff format`
- Check coding style with `uv run ruff check`
- Run tests with `uv run --with pytest pytest`