"""Unit tests for the Count processor."""

import unittest
from unittest.mock import MagicMock, patch

from training_data_packer.processor.count import Count

_MEGATRON_LM = {"default": {"source": "release"}, "fake": {"huggingface": "fake/tokenizer"}}


def _fake_tokenizer(name: str = "fake/tokenizer") -> MagicMock:
    """Create a fake tokenizer that splits text on whitespace.

    :param name: name reported by the tokenizer
    :returns: mocked tokenizer
    """
    tokenizer = MagicMock()
    tokenizer.name_or_path = name
    tokenizer.tokenize.side_effect = lambda text: text.split()
    return tokenizer


class TestCountWithoutTokenizer(unittest.TestCase):
    def setUp(self):
        self.metadata = {"_internal": {}}

    def test_no_tokenizer_loaded(self):
        with patch("training_data_packer.processor.count.get_tokenizer") as get_tokenizer:
            Count(self.metadata)
            get_tokenizer.assert_not_called()

    def test_token_count_false(self):
        metadata = {"_internal": {"token_count": False}, "megatron-lm": _MEGATRON_LM}
        with patch("training_data_packer.processor.count.get_tokenizer") as get_tokenizer:
            counter = Count(metadata)
            get_tokenizer.assert_not_called()
        self.assertNotIn("tokens", counter.get_metrics()["counts"])

    def test_empty_metrics(self):
        counter = Count(self.metadata)
        metrics = counter.get_metrics()
        self.assertEqual(list(metrics.keys()), ["counts"])
        counts = metrics["counts"]
        self.assertEqual(counts["documents"], 0)
        self.assertEqual(counts["segments"], 0)
        self.assertEqual(counts["characters"], 0)
        self.assertEqual(counts["unique_keys"], [])
        self.assertNotIn("tokens", counts)
        self.assertNotIn("tokenizer", counts)

    def test_custom_metric_name(self):
        counter = Count(self.metadata, metric_name="my_counts")
        self.assertEqual(list(counter.get_metrics().keys()), ["my_counts"])

    def test_iterator_counts(self):
        docs = [
            {"id": "1", "text": "Hello world"},
            {"id": "2", "text": "Line one\nLine two\nLine three"},
            {"id": "3", "text": "", "lang": "sv"},
        ]
        counter = Count(self.metadata)
        result = list(counter.get_iterator(iter(docs)))
        self.assertEqual(result, docs)

        counts = counter.get_metrics()["counts"]
        self.assertEqual(counts["documents"], 3)
        self.assertEqual(counts["segments"], 1 + 3 + 1)
        self.assertEqual(counts["characters"], sum(len(d["text"]) for d in docs))
        self.assertEqual(counts["unique_keys"], ["id", "text", "lang"])
        self.assertNotIn("tokens", counts)

    def test_mapper_returns_doc_unchanged(self):
        doc = {"id": "1", "text": "Hello"}
        counter = Count(self.metadata)
        mapper = counter.get_mapper()
        self.assertIs(mapper(doc), doc)
        self.assertEqual(doc, {"id": "1", "text": "Hello"})

    def test_counts_accumulate_across_iterators(self):
        counter = Count(self.metadata)
        list(counter.get_iterator(iter([{"text": "a"}])))
        list(counter.get_iterator(iter([{"text": "bc"}])))
        counts = counter.get_metrics()["counts"]
        self.assertEqual(counts["documents"], 2)
        self.assertEqual(counts["characters"], 3)

    def test_iterator_is_lazy(self):
        counter = Count(self.metadata)
        iterator = counter.get_iterator(iter([{"text": "a"}, {"text": "b"}]))
        self.assertEqual(counter.get_metrics()["counts"]["documents"], 0)
        next(iterator)
        self.assertEqual(counter.get_metrics()["counts"]["documents"], 1)

    def test_missing_text_raises(self):
        counter = Count(self.metadata)
        with self.assertRaises(KeyError):
            list(counter.get_iterator(iter([{"id": "1"}])))


class TestCountWithTokenizer(unittest.TestCase):
    def setUp(self):
        self.metadata = {"_internal": {"token_count": True}, "megatron-lm": _MEGATRON_LM}
        patcher = patch(
            "training_data_packer.processor.count.get_tokenizer",
            return_value=_fake_tokenizer(),
        )
        self.get_tokenizer = patcher.start()
        self.addCleanup(patcher.stop)

    def test_tokenizer_loaded(self):
        Count(self.metadata)
        self.get_tokenizer.assert_called_once_with("fake/tokenizer")

    def test_first_of_several_tokenizers_loaded(self):
        megatron_lm = {
            "default": {"source": "release"},
            "first": {"huggingface": "first/tokenizer"},
            "second": {"huggingface": "second/tokenizer"},
        }
        Count({"_internal": {"token_count": True}, "megatron-lm": megatron_lm})
        self.get_tokenizer.assert_called_once_with("first/tokenizer")

    def test_empty_metrics(self):
        counts = Count(self.metadata).get_metrics()["counts"]
        self.assertEqual(counts["tokens"], 0)
        self.assertEqual(counts["tokenizer"], "fake/tokenizer")

    def test_token_count(self):
        docs = [
            {"id": "1", "text": "Hello world"},
            {"id": "2", "text": "one two three\nfour"},
        ]
        counter = Count(self.metadata)
        list(counter.get_iterator(iter(docs)))
        counts = counter.get_metrics()["counts"]
        self.assertEqual(counts["documents"], 2)
        self.assertEqual(counts["segments"], 3)
        self.assertEqual(counts["tokens"], 6)
        self.assertEqual(counts["characters"], sum(len(d["text"]) for d in docs))

    def test_missing_tokenizer_raises(self):
        metadata = {"_internal": {"token_count": True}}
        with self.assertRaises(ValueError):
            Count(metadata)
        self.get_tokenizer.assert_not_called()

    def test_section_without_tokenizers_raises(self):
        metadata = {"_internal": {"token_count": True}, "megatron-lm": {"default": {"source": "release"}}}
        with self.assertRaises(ValueError):
            Count(metadata)
        self.get_tokenizer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
