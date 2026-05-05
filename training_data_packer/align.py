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
        if "doc_scores" in metadata and metadata["doc_scores"] != "doc_scores":
            if no_key_hierarchy:
                self._mapper["doc_scores"] = [metadata["doc_scores"].split(".")[-1]]
            else:
                self._mapper["doc_scores"] = metadata["doc_scores"].split(".")
        if "web-register" in metadata and metadata["web-register"] != "web-register":
            if no_key_hierarchy:
                self._mapper["web-register"] = [metadata["web-register"].split(".")[-1]]
            else:
                self._mapper["web-register"] = metadata["web-register"].split(".")

    def __iter__(self):
        return self

    def __next__(self):
        src_doc = next(self._src_data)
        for field in self._mapper:
            src_doc[field] = _pop_hierarchy_key_value(self._mapper[field], src_doc)
        return src_doc
