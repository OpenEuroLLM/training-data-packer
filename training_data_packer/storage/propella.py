"""
Query functions for Propella parquet storage.
"""

import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

_FIELD_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")


def query_field(
    directory: str | Path,
    field: str,
    value: Any,
) -> list[dict[str, Any]]:
    """
    Query a directory of parquet files on a given field and return matching records.

    :param directory: Path to directory containing parquet files.
    :param field: Field name to query on (alphanumeric and underscore only).
    :param value: Value to match against the field.
    :return: List of matching records as dicts.
    """
    if not _FIELD_NAME_RE.match(field):
        raise ValueError(f"Invalid field name '{field}': only alphanumeric characters and underscores are allowed.")

    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"'{directory}' is not a valid directory.")

    parquet_files = sorted(dir_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in '{directory}'.")

    result: list[dict[str, Any]] = []
    for pf in parquet_files:
        table = pq.read_table(pf, filters=[(field, "=", value)])
        result.extend(table.to_pylist())

    return result
