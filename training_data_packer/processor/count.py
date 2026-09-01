from collections.abc import Callable, Iterable
from typing import Any

from transformers import AutoTokenizer


def get_tokenizer(tokenizer_name: str):
    return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, use_fast=True)


class Count:
    def __init__(
        self,
        metadata: dict[str, Any],
        metric_name: str = "counts",
    ):
        # Get tokenizer file from factory
        self.tokenizer = get_tokenizer(metadata["tokenizer"])
        self._metric_name = metric_name
        self._documents = 0
        self._segments = 0
        self._tokens = 0
        self._characters = 0
        self._keys = {}

    def get_metrics(self):
        """
        Returns metrics of the mapper.
        :return: Dictionary with metrics.
        """
        return {
            self._metric_name: {
                "documents": self._documents,
                "segments": self._segments,
                "tokens": self._tokens,
                "characters": self._characters,
                "unique_keys": self._keys.keys(),
                "tokenizer": self.tokenizer.name_or_path,
            }
        }

    def get_mapper(self) -> Callable[[list[dict[Any]]], dict[Any]]:

        def mapper(doc: dict[Any]) -> dict[Any]:
            self._documents += 1
            text = doc["text"]
            self._segments += text.count("\n") + 1
            self._tokens += len(self._tokenizer.tokenize(text))
            self._characters += len(text)
            self._keys.update(doc)
            return doc

        return mapper

    def get_iterator(self, data_iterator: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        return map(self.get_mapper(), data_iterator)
