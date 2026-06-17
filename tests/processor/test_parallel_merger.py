import unittest

from training_data_packer.processor.parallel_merger import ParallelLanguageMerger


class TestParallelLanguageMerger(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            "annotations": {
                "parallel": {
                    "src_lang": "src_lang",
                    "source_text": "source_text",
                    "tgt_lang": "tgt_lang",
                    "target_text": "target_text",
                }
            }
        }
        self.merger = ParallelLanguageMerger(self.metadata)

    def test_initialization_default_values(self):
        merger = ParallelLanguageMerger(self.metadata)
        self.assertEqual(merger._src_lang, "src_lang")
        self.assertEqual(merger._source_text_col, "source_text")
        self.assertEqual(merger._tgt_lang, "tgt_lang")
        self.assertEqual(merger._target_text_col, "target_text")
        self.assertEqual(merger._metric_name, "parallel_merger_matching")
        self.assertEqual(merger._processed_records, 0)
        self.assertEqual(merger._written_records, 0)
        self.assertEqual(merger._documents_per_batch, 40)

    def test_initialization_custom_metric_name(self):
        custom_metric = "custom_parallel_merger"
        merger = ParallelLanguageMerger(self.metadata, metric_name=custom_metric)
        self.assertEqual(merger._metric_name, custom_metric)

    def test_initialization_with_missing_metadata_uses_defaults(self):
        metadata = {}
        merger = ParallelLanguageMerger(metadata)
        self.assertEqual(merger._src_lang, "src_lang")
        self.assertEqual(merger._source_text_col, "source_text")
        self.assertEqual(merger._tgt_lang, "tgt_lang")
        self.assertEqual(merger._target_text_col, "target_text")

    def test_get_metrics_initial(self):
        merger = ParallelLanguageMerger(self.metadata)
        metrics = merger.get_metrics()
        expected = {"parallel_merger_matching": {"processed_records": 0, "written_records": 0}}
        self.assertEqual(metrics, expected)

    def test_get_mapper_returns_function(self):
        mapper = self.merger.get_mapper()
        self.assertIsNotNone(mapper)
        self.assertTrue(callable(mapper))

    def test_mapper_single_document(self):
        mapper = self.merger.get_mapper()
        docs = [{"src_lang": "eng", "source_text": "Hello world", "tgt_lang": "fra", "target_text": "Bonjour le monde"}]

        result = mapper(docs)

        self.assertIn("id", result)
        self.assertIn("text", result)
        self.assertEqual("English: Hello world\nFrench: Bonjour le monde", result["text"])
        self.assertIsNotNone(result["id"])

    def test_mapper_multiple_documents(self):
        mapper = self.merger.get_mapper()
        docs = [
            {"src_lang": "eng", "source_text": "Hello", "tgt_lang": "fra", "target_text": "Bonjour"},
            {"src_lang": "eng", "source_text": "Goodbye", "tgt_lang": "deu", "target_text": "Auf Wiedersehen"},
        ]

        result = mapper(docs)

        self.assertIn("id", result)
        self.assertIn("text", result)
        self.assertIn("English: Hello", result["text"])
        self.assertIn("French: Bonjour", result["text"])
        self.assertIn("English: Goodbye", result["text"])
        self.assertIn("German: Auf Wiedersehen", result["text"])
        self.assertIn("\n\n", result["text"])

    def test_mapper_text_format(self):
        mapper = self.merger.get_mapper()
        docs = [
            {"src_lang": "eng", "source_text": "Hello", "tgt_lang": "fra", "target_text": "Bonjour"},
            {"src_lang": "eng", "source_text": "Goodbye", "tgt_lang": "fra", "target_text": "Au revoir"},
        ]

        result = mapper(docs)

        expected_text = "English: Hello\nFrench: Bonjour\n\nEnglish: Goodbye\nFrench: Au revoir"
        self.assertEqual(result["text"], expected_text)

    def test_initialization_custom_metadata_fields(self):
        custom_metadata = {
            "annotations": {
                "parallel": {
                    "src_lang": "source_language",
                    "source_text": "src_text",
                    "tgt_lang": "target_language",
                    "target_text": "tgt_text",
                    "count": 3,
                }
            }
        }

        mapper = ParallelLanguageMerger(custom_metadata)
        self.assertEqual(mapper._src_lang, "source_language")
        self.assertEqual(mapper._source_text_col, "src_text")
        self.assertEqual(mapper._tgt_lang, "target_language")
        self.assertEqual(mapper._target_text_col, "tgt_text")
        self.assertEqual(mapper._documents_per_batch, 3)

    def test_mapper_with_custom_fields(self):
        custom_metadata = {
            "annotations": {
                "parallel": {
                    "src_lang": "source_language",
                    "source_text": "src_text",
                    "tgt_lang": "target_language",
                    "target_text": "tgt_text",
                }
            }
        }

        mapper = ParallelLanguageMerger(custom_metadata).get_mapper()
        docs = [{"source_language": "eng", "src_text": "Hello", "target_language": "fra", "tgt_text": "Bonjour"}]

        result = mapper(docs)

        self.assertIn("English: Hello", result["text"])
        self.assertIn("French: Bonjour", result["text"])

    def test_get_merge_iterator_returns_iterator(self):
        data = [
            {"src_lang": "eng", "source_text": "Hello", "tgt_lang": "fra", "target_text": "Bonjour"},
            {"src_lang": "eng", "source_text": "Goodbye", "tgt_lang": "fra", "target_text": "Au revoir"},
        ]
        result = self.merger.get_merge_iterator(data)
        self.assertTrue(hasattr(result, "__iter__"))

    def test_get_merge_iterator_small_batch(self):
        data = [
            {"src_lang": "eng", "source_text": "Hello", "tgt_lang": "fra", "target_text": "Bonjour"},
            {"src_lang": "eng", "source_text": "Goodbye", "tgt_lang": "fra", "target_text": "Au revoir"},
        ]
        results = list(self.merger.get_merge_iterator(data))
        self.assertEqual(len(results), 1)
        self.assertIn("id", results[0])
        self.assertIn("text", results[0])
        self.assertIn("Hello", results[0]["text"])
        self.assertIn("Goodbye", results[0]["text"])

    def test_get_merge_iterator_with_custom_batch_size(self):
        custom_metadata = {
            "annotations": {
                "parallel": {
                    "src_lang": "src_lang",
                    "source_text": "source_text",
                    "tgt_lang": "tgt_lang",
                    "target_text": "target_text",
                    "count": "2",
                }
            }
        }
        merger = ParallelLanguageMerger(custom_metadata)
        data = [
            {"src_lang": "eng", "source_text": "Hello", "tgt_lang": "fra", "target_text": "Bonjour"},
            {"src_lang": "eng", "source_text": "Goodbye", "tgt_lang": "fra", "target_text": "Au revoir"},
            {"src_lang": "eng", "source_text": "Morning", "tgt_lang": "fra", "target_text": "Bonjour"},
            {"src_lang": "eng", "source_text": "Evening", "tgt_lang": "fra", "target_text": "Bonsoir"},
        ]
        results = list(merger.get_merge_iterator(data))
        self.assertEqual(len(results), 2)

    def test_get_merge_iterator_empty_input(self):
        data = []
        results = list(self.merger.get_merge_iterator(data))
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
