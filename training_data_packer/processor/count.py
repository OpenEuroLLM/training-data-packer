from collections.abc import Callable, Iterable, Iterator
from typing import Any

from loguru import logger
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
        if metadata["_internal"].get("token_count", False):
            try:
                self._tokenizer = get_tokenizer(metadata["tokenizer"])
            except KeyError as e:
                logger.error("Tokenizer is not set")
                raise ValueError("Tokenizer is not set") from e
        else:
            self._tokenizer = None
        self._metric_name = metric_name
        self._documents = 0
        self._segments = 0
        self._tokens = 0
        self._characters = 0
        self._keys = {}

    def get_metrics(self):
        """Get metrics for the mapper.

        :returns: dictionary with metrics
        """
        metrics = {
            "documents": self._documents,
            "segments": self._segments,
            "characters": self._characters,
            "unique_keys": list(self._keys.keys()),
        }
        if self._tokenizer:
            metrics["tokenizer"] = self._tokenizer.name_or_path
            metrics["tokens"] = self._tokens
        return {self._metric_name: metrics}

    def get_mapper(self) -> Callable[[dict[str, Any]], dict[str, Any]]:

        def mapper(doc: dict[str, Any]) -> dict[str, Any]:
            self._documents += 1
            text = doc["text"]
            self._segments += text.count("\n") + 1
            if self._tokenizer:
                self._tokens += len(self._tokenizer.tokenize(text))
            self._characters += len(text)
            self._keys.update(doc)
            return doc

        return mapper

    def get_iterator(self, data_iterator: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        return map(self.get_mapper(), data_iterator)
