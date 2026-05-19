import itertools
from collections.abc import Iterable
from typing import Any


def filter_on_blocklist(data_iterator: Iterable[dict[str, Any]], block_list: list[Any]):
    return itertools.filterfalse(lambda x: x["id"] in block_list, data_iterator)
