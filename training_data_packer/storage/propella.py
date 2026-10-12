"""Query functions for Propella parquet storage."""

import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from loguru import logger

_FIELD_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")


def get_lookup_fn(parquet_path: str | Path, field: str):
    """Get function to query parquet files on a given field and return matching records.

    :param parquet_path: path to a directory containing parquet files, or to one parquet file
    :param field: field name to query on (alphanumeric and underscore only)
    :returns: tuple of the function to look up values, and the metrics of the loaded data
    :raises ValueError: if the field name is invalid
    :raises FileNotFoundError: if no parquet files are found
    """
    if not _FIELD_NAME_RE.match(field):
        raise ValueError(f"Invalid field name '{field}': only alphanumeric characters and underscores are allowed.")

    dir_path = Path(parquet_path)
    if dir_path.is_dir():
        parquet_files = sorted(dir_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in '{parquet_path}'.")
    elif dir_path.is_file():
        parquet_files = [dir_path]
    else:
        raise FileNotFoundError(f"'{parquet_path}' is not a valid directory nor file.")

    lookup_dict: dict[Any, list[dict[str, Any]]] = {}
    logger.info(f"Reading parquet files: {parquet_files}")
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
        """Query in-memory dictionary for matching records.

        :param value: value to match against the field
        :returns: documents matching the value, or None if there is no match
        """
        return lookup_dict.get(value, None)

    return lookup, metrics
