from collections.abc import Callable, Iterable
from typing import Any

from loguru import logger

import training_data_packer.utils.misc
from training_data_packer.metadata import Metadata
from training_data_packer.metadata.defaults import DEFAULT_HASH_ID, DEFAULT_ID, DEFAULT_TEXT


class SourceToPropellaMapper:
    def __init__(
        self,
        metadata: Metadata,
        lookup_fn: Callable[[Any], dict[str, Any]],
        metric_name: str = "propella_matching",
    ):
        """Create SourceToPropellaMapper.

        :param metadata: metadata of the collection, used to find the id, text and hash fields
        :param lookup_fn: function that looks up Propella records by id, returning a list of
            matching records
        :param metric_name: key used for the metrics returned by :meth:`get_metrics`
        """
        self._metric_name = metric_name
        self._processed_records = 0
        self._unmatched_records = 0
        self._multiple_match = 0
        self._lookup_fn = lookup_fn
        self._id_field = metadata.get("id", DEFAULT_ID)
        self._text_field = metadata.get("text", DEFAULT_TEXT)
        self._hash_field = metadata.get("propella-4b.hash-id", DEFAULT_HASH_ID)
        id_hash = metadata.get("propella-4b.hash", None)
        if id_hash is not None:
            self._id_hash_fn = training_data_packer.utils.misc.hash_factory(id_hash)
        else:
            self._id_hash_fn = None

    def get_metrics(self):
        """Get metrics for the mapper.

        :returns: dictionary with metrics
        """
        return {
            self._metric_name: {
                "document_processed": self._processed_records,
                "document_unmatched": self._unmatched_records,
            }
        }

    def get_mapper(self):
        """Get mapper to map source data to Propella records.

        For each document, the mapper looks up the Propella record with the same id using
        ``lookup_fn`` and returns it.

        :returns: function that maps a source document to its Propella record
        """

        def mapper(doc: dict[Any]) -> dict[Any]:
            if self._id_hash_fn is not None:
                id = self._id_hash_fn(doc.get(self._text_field))
            else:
                id = doc.get(self._id_field)
            propella_record = self._lookup_fn(id)
            self._processed_records += 1
            if self._processed_records % 1_000_000 == 0:
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
        """Get metrics for the mapper.

        :returns: dictionary with metrics
        """
        return {
            self._metric_name: {
                "processed_rows": self._processed_rows,
                "rows_with_duplicates": self._duplicates,
                "rows_with_only_id": self._no_match,
            }
        }

    def get_mapper(self):
        """Get a mapper to merge Propella records.

        The mapper aligns multiple records with the same id. It returns the first record
        containing data other than the id, treating subsequent records with data as
        duplicates. It updates counters for processed rows, duplicates and rows with only
        an id.

        :returns: function that maps a list of records to a single record. If no record,
            or more than one record, contains data, a record with only the id is returned.
            The function raises ValueError if the ids of the records do not align.
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
    in_iter: Iterable[dict[str, Any]], propella_data_iter: Iterable[dict[str, Any]] | None
) -> Iterable[dict[str, Any]]:
    """Annotate in_iter with Propella data from propella_data_iter.

    :param in_iter: records to annotate
    :param propella_data_iter: Propella records, aligned with ``in_iter``. If given, its
        length must match ``in_iter``.
    :returns: records with the Propella data merged under ``propella-4b`` where
        applicable, or ``in_iter`` unchanged if ``propella_data_iter`` is None
    """
    if propella_data_iter is not None:

        def mapper(x):
            if "propella-4b" not in x[1]:
                return x[0]
            del x[1]["propella-4b"]["id"]
            return x[0] | x[1]

        return map(mapper, zip(in_iter, propella_data_iter, strict=True))
    return in_iter
