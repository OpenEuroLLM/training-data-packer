from collections.abc import Iterable, Iterator
from typing import Any

import jsonpath_ng

from training_data_packer.metadata import Metadata
from training_data_packer.metadata.defaults import DEFAULT_ID, DEFAULT_TEXT


class AlignFieldNames:
    def __init__(self, src_data: Iterator[Any], metadata: Metadata):
        self._src_data = src_data
        self._mapper = {}
        if "id" in metadata and metadata["id"] != DEFAULT_ID:
            self._mapper["id"] = jsonpath_ng.parse(metadata["id"])
        if "text" in metadata and metadata["text"] != DEFAULT_TEXT:
            self._mapper["text"] = jsonpath_ng.parse(metadata["text"])

    def __iter__(self):
        return self

    def __next__(self):
        src_doc = next(self._src_data)
        for field in self._mapper:
            match = self._mapper[field].find(src_doc)
            src_doc[field] = match[0].value
            self._mapper[field].filter(lambda d: True, src_doc)
        return src_doc


def field_scrubber_factory(data_iterator: Iterable[dict[str, Any]], part_config: dict) -> Iterable[dict[str, Any]]:
    if "scrub" not in part_config or part_config["scrub"] is None or part_config["scrub"] == []:
        return data_iterator
    else:
        scrub_keys = set(part_config["scrub"])
        return map(lambda x: {k: v for k, v in x.items() if k not in scrub_keys}, data_iterator)  # noqa: C417
