import unittest

from parameterized import parameterized

from training_data_packer.processor.clean import AlignFieldNames, field_scrubber_factory
from training_data_packer.utils.metadata import Metadata


class TestAlignFieldNames(unittest.TestCase):
    def test_no_mapping_needed(self):
        src_indata = [
            {"id": "1234", "text": "Happy"},
            {"id": "1235", "text": "Gazonk"},
        ]
        align_it = AlignFieldNames(iter(src_indata), Metadata({"id": "id", "text": "text"}))
        align_list = list(align_it)
        self.assertEqual(align_list, src_indata)

    def test_no_mapping_fields(self):
        src_indata = [
            {"id": "1234", "text": "Happy"},
            {"id": "1235", "text": "Gazonk"},
        ]
        align_it = AlignFieldNames(iter(src_indata), Metadata({}))
        align_list = list(align_it)
        self.assertEqual(align_list, src_indata)

    def test_map_id(self):
        src_indata = [
            {"warcid": "1234", "text": "Happy"},
            {"warcid": "1235", "text": "Gazonk"},
        ]
        expected = [
            {"id": "1234", "text": "Happy"},
            {"id": "1235", "text": "Gazonk"},
        ]
        align_it = AlignFieldNames(iter(src_indata), Metadata({"id": "warcid", "text": "text"}))
        align_list = list(align_it)
        self.assertEqual(align_list, expected)

    def test_map_text(self):
        src_indata = [
            {"id": "1234", "context": "Happy"},
            {"id": "1235", "context": "Gazonk"},
        ]
        expected = [
            {"id": "1234", "text": "Happy"},
            {"id": "1235", "text": "Gazonk"},
        ]
        align_it = AlignFieldNames(iter(src_indata), Metadata({"id": "id", "text": "context"}))
        align_list = list(align_it)
        self.assertEqual(align_list, expected)

    def test_hierarchy_text(self):
        src_indata = [
            {"metadata": {"int-id": "1234"}, "context": "Happy"},
            {"metadata": {"int-id": "1235"}, "context": "Gazonk"},
        ]
        expected = [
            {"id": "1234", "metadata": {}, "text": "Happy"},
            {"id": "1235", "metadata": {}, "text": "Gazonk"},
        ]
        align_it = AlignFieldNames(
            iter(src_indata), Metadata({"id": "metadata.int-id", "text": "context", "doc_s": "doc_score"})
        )
        align_list = list(align_it)
        self.assertEqual(align_list, expected)


class TestScrubFieldNames(unittest.TestCase):
    @parameterized.expand(
        [
            ["no_scrub", [{"id": "1234", "text": "Happy"}], {"sample": "full"}, [{"id": "1234", "text": "Happy"}]],
            [
                "scrub",
                [{"id": "1", "text": "H", "x": "no"}],
                {"sample": "full", "scrub": ["x"]},
                [{"id": "1", "text": "H"}],
            ],
        ]
    )
    def test_no_mapping_needed(self, name, iter, part_config, expected):
        scrub_it = field_scrubber_factory(iter, part_config)
        scrub_list = list(scrub_it)
        self.assertEqual(expected, scrub_list)


if __name__ == "__main__":
    unittest.main()
