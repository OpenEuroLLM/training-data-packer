# AGENTS.md

Packages annotated datasets (PII masking, decontamination, sampling) into final training data.
See `README.md` for the dataset directory layout and data flow, and `docs/` for pipeline and metric details.

## Setup

```bash
uv sync --dev
uv run pre-commit install
```

Python 3.14+ is required (enforced by `pyproject.toml`). Use `uv` for all dependency management; never `pip install`.

## Run

The packer ships four CLI tools. Every tool reads `metadata.yaml` from the collection directory.

| Command                    | Entry point                                  | Purpose                                                        |
|----------------------------|----------------------------------------------|----------------------------------------------------------------|
| `oellm-package-data`       | `training_data_packer/package.py`            | Main tool: lint, sample, release (default) or merge a dataset |
| `oellm-collect-metrics`    | `training_data_packer/collect_metrics.py`    | Collect and summarize metrics from packaged data               |
| `oellm-propella-structure` | `training_data_packer/propella_structure.py` | Arrange Propella data in the same order as the source data     |
| `oellm-propella-merge`     | `training_data_packer/propella_merge.py`     | Merge per-file Propella structure output                       |

Run any of them with `uv run <command> --help`. Common examples:

```bash
uv run oellm-package-data --collection-dir <dir> --mode lint     # validate metadata.yaml
uv run oellm-package-data --collection-dir <dir>                 # release mode: source/ -> release-raw/
uv run oellm-package-data --collection-dir <dir> --mode merge    # release-raw/ -> release/
uv run oellm-collect-metrics --help
```

`lint` mode does not accept `--workers`, `--part`, or `--token-count`.
The `*.sh` scripts in the repo root are SLURM wrappers for the LUMI cluster, so don't run them locally.

## Project layout

- `training_data_packer/mode/` - one module per `oellm-package-data` mode (`lint`, `sample`, `release`, `merge`)
- `training_data_packer/processor/` - streaming processors (clean, filters, PII masking, count, sampling, merge) that wrap record iterators
- `training_data_packer/metadata/` - `metadata.yaml` loading, defaults, and JSON schema validation
- `training_data_packer/storage/` - Propella storage access
- `training_data_packer/utils/` - file, metrics, SLURM, and misc helpers
- `tests/` - mirrors the package layout; fixtures live in `tests/resources/`, integration tests in `tests/integration/`

## Development

Before you finish a change, run:

```bash
uv run ruff format
uv run ruff check --fix
uv run pytest
```

The pre-commit hooks run the same checks (plus trailing-whitespace and end-of-file fixes), and a commit fails if any of them fail.

- Run a single test with `uv run pytest tests/processor/test_count.py` or `uv run pytest -k <name>`.
- Write tests with `unittest.TestCase`. Use `parameterized` for table-driven cases, following the existing tests.
- Add tests for new features and bug fixes.
- Log with `loguru` (`from loguru import logger`), not `print`.
- When you add or rename metrics, update `docs/metrics.md`.

## Code style

Ruff enforces formatting and most lint rules (config in `pyproject.toml`):

- PEP 8, max line length 120
- Double quotes (`"`) and spaces for indentation
- Magic trailing commas: when a call or literal is split over several lines, end it with a trailing comma
- End every file with a newline
- Prefix private functions, methods, and attributes with an underscore (`_`)
- Use modern type hints (`list[str]`, `X | None`, and `collections.abc` types)

## Documentation

- Write docstrings for modules, classes, and functions.
- Write the first line in the imperative mood ("Count tokens", not "Counts tokens").
- Use reST style:

```python
def scale(value: int, factor: int) -> int:
    """Scale a value by a factor.

    :param value: the value to scale
    :param factor: the multiplier
    :returns: the scaled value
    :raises ValueError: if factor is negative
    """
```
