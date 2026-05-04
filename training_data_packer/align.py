from typing import List, Any, Dict


def _pop_hierarchy_key_value(key: List[Any], data: Dict) -> Any:
    value = data[key[0]]
    if key[1:]:
        value = _pop_hierarchy_key_value(key[1:], value)
    else:
        del data[key[0]]
    return value

class AlignFieldNames:

    def __init__(self, src_data, metadata: Dict, no_key_hierarchy = False):
        self._src_data = src_data
        self._mapper = {}
        if "id" in metadata and metadata["id"]!="id":
            if no_key_hierarchy:
                self._mapper["id"] = [metadata["id"].split(".")[-1]]
            else:
                self._mapper["id"] = metadata["id"].split(".")
        if "text" in metadata and metadata["text"]!="text":
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
