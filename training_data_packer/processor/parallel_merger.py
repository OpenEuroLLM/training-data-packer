from typing import Any

import training_data_packer.utils.misc
from training_data_packer.utils.metadata import get_metadata_value


class ParallelLanguageMerger:
    def __init__(
        self,
        metadata: dict[str, Any],
        metric_name: str = "parallel_merger_matching",
    ):
        self._metric_name = metric_name
        self._processed_records = 0
        self._written_records = 0
        self._hash_fn = training_data_packer.hash_factory("sha256")
        self._src_lang = get_metadata_value(metadata, "annotations.parallel.src_lang", "src_lang")
        self._source_text_col = get_metadata_value(metadata, "annotations.parallel.source_text", "source_text")
        self._tgt_lang = get_metadata_value(metadata, "annotations.parallel.tgt_lang", "tgt_lang")
        self._target_text_col = get_metadata_value(metadata, "annotations.parallel.target_text", "target_text")

    def get_metrics(self):
        """
        Returns metrics of the mapper.
        :return: Dictionary with metrics.
        """
        return {
            self._metric_name: {"processed_records": self._processed_records, "written_records": self._written_records}
        }

    def get_mapper(self):

        def mapper(docs: list[dict[Any]]) -> dict[Any]:
            formatted_pairs = []
            for doc in docs:
                src_lang_name = training_data_packer.lang_to_name(doc[self._src_lang])
                target_lang_name = training_data_packer.lang_to_name(doc[self._src_lang])
                pair = (
                    f"{src_lang_name}: {doc[self._source_text_col]}\n{target_lang_name}: {doc[self._target_text_col]}"
                )
                formatted_pairs.append(pair)
            final_text = "\n\n".join(formatted_pairs)
            identity = self._hash_fn(final_text)
            doc = {"id": identity, "text": final_text}
            return doc

        return mapper
