import itertools
from collections.abc import Iterable
from typing import Any


class FilterOnBlocklist:
    """Filter out documents with ids in a blocklist.

    :param name: name of the filter, used as key in the metrics
    :param block_list: ids of documents to filter out
    """

    def __init__(self, name: str, block_list: set[Any]):
        self.counter = 0
        self.name = name
        self.block_list = {str(x) for x in block_list}

    def get_metrics(self):
        """Get metrics for the filter.

        :returns: dictionary with metrics
        """
        return {
            self.name: {
                "document_removed": self.counter,
                "blocklist_length": len(self.block_list),
            }
        }

    def filter(self, data_iterator: Iterable[dict[str, Any]]):
        """Filter out documents with ids in blocklist.

        :param data_iterator: iterator over documents
        :returns: iterator over filtered documents
        """

        def filter_fn(x: dict[str, Any]) -> bool:
            if str(x["id"]) in self.block_list:
                self.counter += 1
                return True
            else:
                return False

        return itertools.filterfalse(filter_fn, data_iterator)
