import itertools
import random
from collections.abc import Callable, Iterable
from typing import Any

from training_data_packer.metadata import Metadata
from training_data_packer.metadata.defaults import (
    PARALLEL_COUNT_DEFAULT,
    SRC_LANGUAGE_DEFAULT,
    SRC_TEXT_DEFAULT,
    TGT_LANGUAGE_DEFAULT,
    TGT_TEXT_DEFAULT,
)
from training_data_packer.utils.misc import get_dict_value, hash_factory, lang_to_name


class ParallelLanguageMerger:
    """Merge parallel language documents.

    Merge batches of parallel language documents into a unified text format with
    unique identification. Source and target language pairs are formatted, and the
    order within each pair is optionally randomized. A unique ID is calculated for
    the merged content using a hash function. Processing metrics, such as the count
    of processed and written records, are tracked.

    :param metadata: metadata of the collection
    :param part_config: configuration of the part being merged
    :param flip_fn: function deciding whether to flip the order of a pair; defaults to
        a random flip with the probability configured in ``parallel.flip``
    :param metric_name: key used for the metrics
    """

    def __init__(
        self,
        metadata: Metadata,
        part_config: dict[str, Any],
        flip_fn: Callable[[], bool] | None = None,
        metric_name: str = "parallel_merger_matching",
    ):
        self._metric_name = metric_name
        self._processed_records = 0
        self._written_records = 0
        flip_probability = get_dict_value(part_config, "parallel.flip", default=0.5)
        self._flip_fn = flip_fn or (lambda: random.random() < flip_probability)
        self._hash_fn = hash_factory("sha256")
        self._src_lang = metadata.get("parallel.source.language", SRC_LANGUAGE_DEFAULT)
        self._source_text_col = metadata.get("parallel.source.text", SRC_TEXT_DEFAULT)
        self._tgt_lang = metadata.get("parallel.target.language", TGT_LANGUAGE_DEFAULT)
        self._target_text_col = metadata.get("parallel.target.text", TGT_TEXT_DEFAULT)
        self._documents_per_batch = get_dict_value(part_config, "parallel.count", default=PARALLEL_COUNT_DEFAULT)

    def get_metrics(self):
        """Get metrics for the mapper.

        :returns: dictionary with metrics
        """
        return {
            self._metric_name: {
                "document_processed": self._processed_records,
                "document_written": self._written_records,
            }
        }

    def get_mapper(self) -> Callable[[list[dict[Any]]], dict[Any]]:
        """Get mapper to merge language pairs.

        The mapper transforms a list of documents into a single document. It concatenates
        source and target language texts with their language names, and computes a unique
        id for the resulting content using a hash function.

        :returns: function that maps a list of documents to a single dictionary with the
            formatted text and a unique id
        """

        def mapper(docs: list[dict[Any]]) -> dict[Any]:
            formatted_pairs = []
            flip_order = self._flip_fn()
            for doc in docs:
                src_lang_name = lang_to_name(doc[self._src_lang])
                target_lang_name = lang_to_name(doc[self._tgt_lang])
                if flip_order:
                    pair = (
                        f"{target_lang_name}: {doc[self._target_text_col]}\n"
                        f"{src_lang_name}: {doc[self._source_text_col]}"
                    )
                else:
                    pair = (
                        f"{src_lang_name}: {doc[self._source_text_col]}\n"
                        f"{target_lang_name}: {doc[self._target_text_col]}"
                    )
                formatted_pairs.append(pair)
                self._processed_records += 1
            final_text = "\n\n".join(formatted_pairs)
            identity = self._hash_fn(final_text)
            doc = {"id": identity, "text": final_text}
            self._written_records += 1
            return doc

        return mapper

    def get_merge_iterator(self, data_iterator: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        return map(self.get_mapper(), itertools.batched(data_iterator, self._documents_per_batch, strict=False))


class ParallelSyntheticId:
    """Create a synthetic ID of a language pair.

    :param metadata: metadata of the collection
    :param metric_name: key used for the metrics
    """

    def __init__(
        self,
        metadata: Metadata,
        metric_name: str = "parallel_synthetic_id",
    ):
        self._metric_name = metric_name
        self._processed_records = 0
        hash_fn = hash_factory("sha256")
        source_text_col = metadata.get("parallel.source.text", "source_text")
        target_text_col = metadata.get("parallel.target.text", "target_text")
        self._id_fn = lambda doc: hash_fn(f"{doc[source_text_col]}{doc[target_text_col]}")

    def get_metrics(self):
        """Get metrics for the mapper.

        :returns: dictionary with metrics
        """
        return {self._metric_name: {"document_processed": self._processed_records}}

    def get_mapper(self) -> Callable[[dict[str, Any]], dict[str, Any]]:

        def mapper(doc: dict[str, Any]) -> dict[str, Any]:
            doc["id"] = self._id_fn(doc)
            self._processed_records += 1
            return doc

        return mapper

    def get_iterator(self, data_iterator: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        return map(self.get_mapper(), data_iterator)
