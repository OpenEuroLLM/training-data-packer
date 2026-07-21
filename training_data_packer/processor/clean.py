from collections.abc import Iterable, Iterator
from typing import Any

import glom


class AlignFieldNames:
    def __init__(self, src_data: Iterator[Any], metadata: dict):
        self._src_data = src_data
        self._mapper = {}
        if "id" in metadata and metadata["id"] != "id":
            self._mapper["id"] = metadata["id"]
        if "text" in metadata and metadata["text"] != "text":
            self._mapper["text"] = metadata["text"]

    def __iter__(self):
        return self

    def __next__(self):
        src_doc = next(self._src_data)
        for field in self._mapper:
            try:
                key = self._mapper[field]
                key = key.replace("[", ".").replace("]", "")
                src_doc[field] = glom.glom(src_doc, key)
                glom.delete(src_doc, key)
            except glom.PathAccessError as e:
                if field not in src_doc:
                    raise e
        return src_doc


def field_scrubber_factory(data_iterator: Iterable[dict[str, Any]], part_config: dict) -> Iterable[dict[str, Any]]:
    if "scrub" not in part_config or part_config["scrub"] is None or part_config["scrub"] == []:
        return data_iterator
    else:
        scrub_keys = set(part_config["scrub"])
        return map(lambda x: {k: v for k, v in x.items() if k not in scrub_keys}, data_iterator)  # noqa: C417
