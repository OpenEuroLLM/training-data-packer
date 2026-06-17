import itertools
from collections.abc import Callable, Iterable
from typing import Any

from training_data_packer.utils.metadata import get_metadata_value
from training_data_packer.utils.misc import hash_factory, lang_to_name


class ParallelLanguageMerger:
    def __init__(
        self,
        metadata: dict[str, Any],
        metric_name: str = "parallel_merger_matching",
    ):
        self._metric_name = metric_name
        self._processed_records = 0
        self._written_records = 0
        self._hash_fn = hash_factory("sha256")
        self._src_lang = get_metadata_value(metadata, "annotations.parallel.src_lang", "src_lang")
        self._source_text_col = get_metadata_value(metadata, "annotations.parallel.source_text", "source_text")
        self._tgt_lang = get_metadata_value(metadata, "annotations.parallel.tgt_lang", "tgt_lang")
        self._target_text_col = get_metadata_value(metadata, "annotations.parallel.target_text", "target_text")
        self._documents_per_batch = int(get_metadata_value(metadata, "annotations.parallel.count", "40"))

    def get_metrics(self):
        """
        Returns metrics of the mapper.
        :return: Dictionary with metrics.
        """
        return {
            self._metric_name: {"processed_records": self._processed_records, "written_records": self._written_records}
        }

    def get_mapper(self) -> Callable[[list[dict[Any]]], dict[Any]]:
        """
        Constructs and returns a processing function designed to transform a list
        of raw document dictionaries into a standardized format. The generated
        function concatenates source and target language texts with their respective
        language names, generates a unified text block from these pairs, and
        computes a unique identifier for the resulting content based on a hash
        function.

        :return: A callable object that processes a list of documents and returns a
            single dictionary containing the formatted text and a unique ID.
        :rtype: Callable[[list[dict[Any]]], dict[Any]]
        """

        def mapper(docs: list[dict[Any]]) -> dict[Any]:
            formatted_pairs = []
            for doc in docs:
                src_lang_name = lang_to_name(doc[self._src_lang])
                target_lang_name = lang_to_name(doc[self._tgt_lang])
                pair = (
                    f"{src_lang_name}: {doc[self._source_text_col]}\n{target_lang_name}: {doc[self._target_text_col]}"
                )
                formatted_pairs.append(pair)
            final_text = "\n\n".join(formatted_pairs)
            identity = self._hash_fn(final_text)
            doc = {"id": identity, "text": final_text}
            return doc

        return mapper

    def get_merge_iterator(self, data_iterator: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        return map(self.get_mapper(), itertools.batched(data_iterator, self._documents_per_batch, strict=False))
