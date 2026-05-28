from collections.abc import Callable
from typing import Any

from loguru import logger


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

    def __init__(self, id_field: str, lookup_fn: Callable[[Any], dict[str, Any]], metric_name: str = "doc_ordering"):
        self._metric_name = metric_name
        self._processed_records = 0
        self._unmatched_records = 0
        self._multiple_match = 0
        self._lookup_fn = lookup_fn
        self._id_field = id_field

    def get_metrics(self):
        """
        Returns metrics of the mapper.
        :return: Dictionary with metrics.
        """
        return {
            self._metric_name: {
                "processed_records": self._processed_records,
                "unmatched_records": self._unmatched_records,
                "multiple_match": self._multiple_match,
            }
        }

    def get_mapper(self):
        """
        Returns a mapper that for each element it get lookup the object with same id
        using lookup_fn and returns it
        :return: function lookup objects
        """

        def mapper(doc: dict[Any]) -> dict[Any]:
            id = doc.get(self._id_field)
            propella_records = self._lookup_fn(id)
            len_records = len(propella_records)
            self._processed_records += 1
            if self._processed_records % 1000 == 0:
                logger.info(f"{self._processed_records} records processed")
            if len_records == 1:
                return propella_records[0]
            elif len_records == 0:
                self._unmatched_records += 1
                return {self._id_field: id}
            else:
                logger.warning(f"Lookup function returns multiple answers for {id}, using the first found.")
                self._multiple_match += 1
                return propella_records[0]

        return mapper
