import unittest

from training_data_packer.processor.parallel_merger import ParallelLanguageMerger, ParallelSyntheticId


class TestParallelLanguageMerger(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            "parallel": {
                "source": {"language": "src_lang", "text": "source_text"},
                "target": {"language": "tgt_lang", "text": "target_text"},
            }
        }
        self.part_config = {"parallel": {"count": "40"}}

    def test_initialization_default_values(self):
        merger = ParallelLanguageMerger(self.metadata, self.part_config)
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
        merger = ParallelLanguageMerger(self.metadata, self.part_config, metric_name=custom_metric)
        self.assertEqual(merger._metric_name, custom_metric)

    def test_initialization_with_missing_metadata_uses_defaults(self):
        metadata = {}
        merger = ParallelLanguageMerger(metadata, {})
        self.assertEqual(merger._src_lang, "src_lang")
        self.assertEqual(merger._source_text_col, "source_text")
        self.assertEqual(merger._tgt_lang, "tgt_lang")
        self.assertEqual(merger._target_text_col, "target_text")
        self.assertEqual(merger._documents_per_batch, 40)

    def test_get_metrics_initial(self):
        merger = ParallelLanguageMerger(self.metadata, {})
        metrics = merger.get_metrics()
        expected = {"parallel_merger_matching": {"processed_records": 0, "written_records": 0}}
        self.assertEqual(metrics, expected)

    def test_mapper_single_document(self):
        merger = ParallelLanguageMerger(self.metadata, {})
        mapper = merger.get_mapper()
        docs = [{"src_lang": "eng", "source_text": "Hello world", "tgt_lang": "fra", "target_text": "Bonjour le monde"}]

        result = mapper(docs)

        self.assertIn("id", result)
        self.assertIn("text", result)
        self.assertIn(
            result["text"],
            [
                "English: Hello world\nFrench: Bonjour le monde",
                "French: Bonjour le monde\nEnglish: Hello world",
            ],
        )
        self.assertIsNotNone(result["id"])

    def test_mapper_multiple_documents(self):
        merger = ParallelLanguageMerger(self.metadata, {})
        mapper = merger.get_mapper()
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
        merger = ParallelLanguageMerger(self.metadata, {}, flip_fn=lambda: False)
        mapper = merger.get_mapper()
        docs = [
            {"src_lang": "eng", "source_text": "Hello", "tgt_lang": "fra", "target_text": "Bonjour"},
            {"src_lang": "eng", "source_text": "Goodbye", "tgt_lang": "fra", "target_text": "Au revoir"},
        ]

        result = mapper(docs)

        expected_text = "English: Hello\nFrench: Bonjour\n\nEnglish: Goodbye\nFrench: Au revoir"
        self.assertIn(expected_text, result["text"])

    def test_initialization_custom_metadata_fields(self):
        custom_metadata = {
            "parallel": {
                "source": {"language": "source_language", "text": "src_text"},
                "target": {"language": "target_language", "text": "tgt_text"},
            }
        }
        custom_part_config = {"parallel": {"count": "3"}}

        mapper = ParallelLanguageMerger(custom_metadata, custom_part_config)
        self.assertEqual(mapper._src_lang, "source_language")
        self.assertEqual(mapper._source_text_col, "src_text")
        self.assertEqual(mapper._tgt_lang, "target_language")
        self.assertEqual(mapper._target_text_col, "tgt_text")
        self.assertEqual(mapper._documents_per_batch, 3)

    def test_mapper_with_custom_fields(self):
        custom_metadata = {
            "parallel": {
                "source": {"language": "source_language", "text": "src_text"},
                "target": {"language": "target_language", "text": "tgt_text"},
            }
        }

        mapper = ParallelLanguageMerger(custom_metadata, self.part_config).get_mapper()
        docs = [{"source_language": "eng", "src_text": "Hello", "target_language": "fra", "tgt_text": "Bonjour"}]

        result = mapper(docs)

        self.assertIn("English: Hello", result["text"])
        self.assertIn("French: Bonjour", result["text"])

    def test_get_merge_iterator_small_batch(self):
        data = [
            {"src_lang": "eng", "source_text": "Hello", "tgt_lang": "fra", "target_text": "Bonjour"},
            {"src_lang": "eng", "source_text": "Goodbye", "tgt_lang": "fra", "target_text": "Au revoir"},
        ]
        results = list(ParallelLanguageMerger(self.metadata, {}).get_merge_iterator(data))
        self.assertEqual(len(results), 1)
        self.assertIn("id", results[0])
        self.assertIn("text", results[0])
        self.assertIn("Hello", results[0]["text"])
        self.assertIn("Goodbye", results[0]["text"])

    def test_get_merge_iterator_with_custom_batch_size(self):
        custom_part_config = {"parallel": {"count": "2"}}
        merger = ParallelLanguageMerger(self.metadata, custom_part_config)
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
        results = list(ParallelLanguageMerger(self.metadata, self.part_config).get_merge_iterator(data))
        self.assertEqual(len(results), 0)


class TestParallelSyntheticId(unittest.TestCase):
    def test_create_synthetic_id_mapper_default_config(self):
        doc = {"source_text": "foo bar", "target_text": "gazonk"}
        to_test = ParallelSyntheticId({})
        result = to_test.get_mapper()(doc)
        self.assertEqual("ecefffc59192dd7f300f750650a3d21e7a2ea1c9854552ab2f83c7fb4986a08b", result["id"])
        self.assertEqual({"parallel_synthetic_id": {"processed_records": 1}}, to_test.get_metrics())

    def test_create_synthetic_id_mapper_alternativ_fields(self):
        doc = {"field1": "foo bar", "field2": "gazonk"}
        metadata = {
            "parallel": {
                "source": {
                    "text": "field1",
                },
                "target": {
                    "text": "field2",
                },
            }
        }
        to_test = ParallelSyntheticId(metadata)
        result = to_test.get_mapper()(doc)
        self.assertEqual("ecefffc59192dd7f300f750650a3d21e7a2ea1c9854552ab2f83c7fb4986a08b", result["id"])
        self.assertEqual({"parallel_synthetic_id": {"processed_records": 1}}, to_test.get_metrics())


if __name__ == "__main__":
    unittest.main()
