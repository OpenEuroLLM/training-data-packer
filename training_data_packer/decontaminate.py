from loguru import logger


class Decontaminate:
    def __init__(self, src_data, decontamination_data):
        self._src_data = src_data
        self._decontamination_data = decontamination_data
        self._next_doc_to_remove = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            if self._next_doc_to_remove is None:
                self._next_doc_to_remove = next(self._decontamination_data)["id"]
        except StopIteration:
            self._next_doc_to_remove = None
        try:
            next_src_doc = next(self._src_data)
        except StopIteration as e:
            if self._next_doc_to_remove is not None:
                logger.error(f"Decontamination iteration failed, docid left {self._next_doc_to_remove}")
                raise ValueError() from e
            raise e
        if self._next_doc_to_remove is not None and self._next_doc_to_remove == next_src_doc["id"]:
            next_src_doc["delete"] = True
            try:
                self._next_doc_to_remove = next(self._decontamination_data)["id"]
            except StopIteration:
                self._next_doc_to_remove = None
        return next_src_doc
