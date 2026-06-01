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

    def __init__(
        self, id_field: str, lookup_fn: Callable[[Any], dict[str, Any]], metric_name: str = "propella_matching"
    ):
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
            propella_record = self._lookup_fn(id)
            self._processed_records += 1
            if self._processed_records % 100_000 == 0:
                logger.info(f"{self._processed_records} records processed")
            if propella_record is None:
                self._unmatched_records += 1
                return {self._id_field: id}
            else:
                return propella_record

        return mapper
