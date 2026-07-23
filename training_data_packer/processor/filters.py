import itertools
from collections.abc import Iterable
from typing import Any


class FilterOnBlocklist:
    """
    Filter for blocklist. Filters out documents with ids in blocklist.
    """

    def __init__(self, name: str, block_list: set[Any]):
        self.counter = 0
        self.name = name
        self.block_list = {str(x) for x in block_list}

    def get_metrics(self):
        """
        Returns metrics of the filter.
        :return: Dictionary with metrics.
        """
        return {
            self.name: {
                "removed": self.counter,
                "list_length": len(self.block_list),
            }
        }

    def filter(self, data_iterator: Iterable[dict[str, Any]]):
        """
        Filters out documents with ids in blocklist.
        :param data_iterator: Iterator over documents.
        :return: Iterator over filtered documents.
        """

        def filter_fn(x: dict[str, Any]) -> bool:
            if str(x["id"]) in self.block_list:
                self.counter += 1
                return True
            else:
                return False

        return itertools.filterfalse(filter_fn, data_iterator)
