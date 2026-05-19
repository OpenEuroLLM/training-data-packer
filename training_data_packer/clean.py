from collections.abc import Iterable
from typing import Any


def _pop_hierarchy_key_value(key: list[Any], data: dict) -> Any:
    value = data[key[0]]
    if key[1:]:
        value = _pop_hierarchy_key_value(key[1:], value)
    else:
        del data[key[0]]
    return value


class AlignFieldNames:
    def __init__(self, src_data, metadata: dict, no_key_hierarchy=False):
        self._src_data = src_data
        self._mapper = {}
        if "id" in metadata and metadata["id"] != "id":
            if no_key_hierarchy:
                self._mapper["id"] = [metadata["id"].split(".")[-1]]
            else:
                self._mapper["id"] = metadata["id"].split(".")
        if "text" in metadata and metadata["text"] != "text":
            if no_key_hierarchy:
                self._mapper["text"] = [metadata["text"].split(".")[-1]]
            else:
                self._mapper["text"] = metadata["text"].split(".")

    def __iter__(self):
        return self

    def __next__(self):
        src_doc = next(self._src_data)
        for field in self._mapper:
            src_doc[field] = _pop_hierarchy_key_value(self._mapper[field], src_doc)
        return src_doc


def field_scrubber_factory(data_iterator: Iterable[dict[str, Any]], part_config: dict) -> Iterable[dict[str, Any]]:
    if "scrub" not in part_config or part_config["scrub"] is None or part_config["scrub"] == []:
        return data_iterator
    else:
        scrub_keys = set(part_config["scrub"])
        return map(lambda x: {k: v for k, v in x.items() if k not in scrub_keys}, data_iterator)  # noqa: C417
