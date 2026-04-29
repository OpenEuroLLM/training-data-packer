

class AlignFieldNames:

    def __init__(self, src_data, metadata):
        self._src_data = src_data
        self._mapper = {}
        if "id" in metadata and metadata["id"]!="id":
            self._mapper[metadata["id"]] = "id"
        if "text" in metadata and metadata["text"]!="text":
            self._mapper[metadata["text"]] = "text"

    def __iter__(self):
        return self

    def __next__(self):
        next_src_doc = next(self._src_data)
        for in_field in self._mapper:
            if in_field in next_src_doc:
                next_src_doc[self._mapper[in_field]] = next_src_doc[in_field]
                del next_src_doc[in_field]
        return next_src_doc
