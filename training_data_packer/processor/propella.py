from collections.abc import Callable, Iterable
from typing import Any

from loguru import logger

import training_data_packer.utils.misc
from training_data_packer.utils.metadata import get_metadata_value


class SourceToPropellaMapper:
    """
    Initializes the SourceToPropellaMapper with the provided metadata,
    lookup function, and metric name.

    :param id_field: Name of the id field.
    :param lookup_fn: Function responsible for looking up Propella records
        based on an ID. It accepts an ID and returns a list of matching
        records.
    :param metric_name: The key name used to store the metrics in the
        dictionary returned by the get_metrics method.
    """

    def __init__(
        self,
        metadata: dict[str, Any],
        lookup_fn: Callable[[Any], dict[str, Any]],
        metric_name: str = "propella_matching",
    ):
        self._metric_name = metric_name
        self._processed_records = 0
        self._unmatched_records = 0
        self._multiple_match = 0
        self._lookup_fn = lookup_fn
        self._id_field = metadata["id"]
        self._text_field = metadata["text"]
        self._hash_field = get_metadata_value(metadata, "annotations.propella-4b.hash-id", "hash")
        id_hash = get_metadata_value(metadata, "annotations.propella-4b.hash", None)
        if id_hash is not None:
            self._id_hash_fn = training_data_packer.utils.misc.hash_factory(id_hash)
        else:
            self._id_hash_fn = None

    def get_metrics(self):
        """
        Returns metrics of the mapper.
        :return: Dictionary with metrics.
        """
        return {
            self._metric_name: {
                "processed_records": self._processed_records,
                "unmatched_records": self._unmatched_records,
            }
        }

    def get_mapper(self):
        """
        Returns a mapper that for each element it get lookup the object with same id
        using lookup_fn and returns it
        :return: function lookup objects
        """

        def mapper(doc: dict[Any]) -> dict[Any]:
            if self._id_hash_fn is not None:
                id = self._id_hash_fn(doc.get(self._text_field))
            else:
                id = doc.get(self._id_field)
            propella_record = self._lookup_fn(id)
            self._processed_records += 1
            if self._processed_records % 100_000 == 0:
                logger.info(f"{self._processed_records} records processed")
            if propella_record is None:
                self._unmatched_records += 1
                if self._id_hash_fn is not None:
                    return {"id": doc.get(self._id_field), self._hash_field: id}
                else:
                    return {"id": id}
            else:
                result_doc = {"id": doc.get(self._id_field), "propella-4b": propella_record}
                if self._id_hash_fn:
                    result_doc[self._hash_field] = propella_record["id"]
                return result_doc

        return mapper


class MergePropellaRecords:
    def __init__(self, id_field: str, metric_name: str = "propella_merge"):
        self._metric_name = metric_name
        self._id_field = id_field
        self._processed_rows = 0
        self._duplicates = 0
        self._no_match = 0

    def get_metrics(self):
        """
        Returns metrics of the mapper.
        :return: Dictionary with metrics.
        """
        return {
            self._metric_name: {
                "processed_rows": self._processed_rows,
                "rows_with_duplicates": self._duplicates,
                "rows_with_only_id": self._no_match,
            }
        }

    def get_mapper(self):
        """
        Generates and returns a mapper function designed to process and align multiple
        document entries based on an id. The returned function validates
        that all provided dictionaries reference the same entity ID. It iterates
        through the input to locate the first document containing data beyond the
        identifier key, treating subsequent data sources as duplicates. The mapper
        updates internal tracking counters for processed rows, duplicates, and
        non-matching entries during execution.

        :return: A callable that takes a list of dictionaries and returns a single
            dictionary. The function validates ID alignment across the input list. It
            returns the first record found containing data other than the ID. If
            multiple records contain data, it returns a dictionary with only the ID.
            If no data is found, it returns a dictionary with only the ID. It raises
            a ValueError if the identifiers in the documents do not align.
        :rtype: Callable[[list[dict[Any]]], dict[Any]]
        """

        def mapper(docs: list[dict[Any]]) -> dict[Any]:
            id_value = docs[0].get(self._id_field)
            self._processed_rows += 1
            candidate = None
            duplicate = False
            for d in docs:
                if d[self._id_field] != id_value:
                    raise ValueError("Files not aligned")
                if "propella-4b" in d:
                    if candidate is None:
                        candidate = d
                    else:
                        duplicate = True
                        self._duplicates += 1
                        break
            if duplicate:
                result = {self._id_field: id_value}
                if "hash" in docs[0]:
                    result["hash"] = docs[0]["hash"]
                return result
            if candidate is None:
                self._no_match += 1
                result = {self._id_field: id_value}
                if "hash" in docs[0]:
                    result["hash"] = docs[0]["hash"]
                return result
            return candidate

        return mapper


def propella_annotate_factory(
    in_iter: Iterable[dict[str, Any]], propella_data_iter: None | Iterable[dict[str, Any]]
) -> Iterable[dict[str, Any]]:
    """
    Annotate in_iter with propella data from propella_data_iter.

    :param in_iter: An iterable of dictionaries representing the primary data
        objects requiring potential annotation.
    :param propella_data_iter: An optional iterable of dictionaries sourced from
        Propella. If provided, its length must match the input iterable to allow
        for strict alignment.

    :return: An iterable of dictionaries that includes the merged Propella data
        under 'propella-4b' where applicable, or the original data if the Propella
        iterator is None.
    """
    if propella_data_iter is not None:

        def mapper(x):
            if len(x[1]) == 1:
                return x[0]
            del x[1]["id"]
            x[0]["propella-4b"] = x[1]
            return x[0]

        return map(mapper, zip(in_iter, propella_data_iter, strict=True))
    return in_iter
