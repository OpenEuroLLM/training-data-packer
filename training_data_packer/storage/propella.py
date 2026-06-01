"""
Query functions for Propella parquet storage.
"""

import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from loguru import logger

_FIELD_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")


def get_lookup_fn(directory: str | Path, field: str):
    """
    Creates a function query a directory of parquet files on a given field
    and return matching records.

    :param directory: Path to directory containing parquet files.
    :param field: Field name to query on (alphanumeric and underscore only).
    :return: Function to lookup values in parquet files.
    """

    if not _FIELD_NAME_RE.match(field):
        raise ValueError(f"Invalid field name '{field}': only alphanumeric characters and underscores are allowed.")

    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"'{directory}' is not a valid directory.")

    parquet_files = sorted(dir_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in '{directory}'.")

    lookup_dict: dict[Any, list[dict[str, Any]]] = {}
    logger.info("Reading parquet files: {','.join(parquet_files)")
    duplicated_keys = set()
    rows_read = 0
    missing_id_field = 0
    for pf in parquet_files:
        parquet_file = pq.ParquetFile(pf)
        for k, record_batch in enumerate(parquet_file.iter_batches(batch_size=10_000)):
            logger.info(f"Processing batch number: {k}")
            for record in record_batch.to_pylist():
                rows_read += 1
                if field in record:
                    key = record[field]
                    if key in lookup_dict:
                        duplicated_keys.add(key)
                        # logger.warning(f"Key {key} is duplicated in propella")
                    else:
                        lookup_dict[key] = record
                else:
                    missing_id_field += 1
    logger.info("Done reading parquet files.")
    for k in duplicated_keys:  # Deleting duplicates after inserted all to avoid keys reappearing
        del lookup_dict[k]
    unique_keys = len(lookup_dict)
    duplicated_keys_count = len(duplicated_keys)
    duplicated_keys = None
    logger.info(f"Duplicated keys in propella: {duplicated_keys_count} unique keys: {unique_keys}")
    metrics = {
        "propella_src": {
            "duplicated_keys": duplicated_keys_count,
            "unique_keys": unique_keys,
            "rows_read": rows_read,
            "missing_id_field": missing_id_field,
        }
    }

    def lookup(value: Any) -> list[dict[str, Any]]:
        """
        Query in-memory dictionary for matching records.
        :param value: Value to match against field.
        :return: List of documents matching value.
        """
        return lookup_dict.get(value, None)

    return lookup, metrics
