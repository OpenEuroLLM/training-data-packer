import unittest

from training_data_packer import align
from training_data_packer.align import AlignFieldNames


class TestGetHierarchyKey(unittest.TestCase):

    def test_pop_key_one_level(self):
        input_data = {"id": "1234", "text": "Happy"}
        result = align._pop_hierarchy_key_value(["id"], input_data)
        self.assertEqual(result, "1234")
        self.assertEqual({"text": "Happy"}, input_data)

    def test_pop_key_two_level(self):
        input_data = {"metadata": {"id": "1234", "text": "Happy"}}
        result = align._pop_hierarchy_key_value(["metadata", "id"], input_data)
        self.assertEqual(result, "1234")
        self.assertEqual({"metadata": {"text": "Happy"}}, input_data)

class TestAlignFieldNames(unittest.TestCase):
    def test_no_mapping_needed(self):
        src_indata = [
            {"id": "1234", "text": "Happy"},
            {"id": "1235", "text": "Gazonk"},
        ]
        align_it = AlignFieldNames(iter(src_indata), {"id": "id", "text": "text"})
        align_list = list(align_it)
        self.assertEqual(align_list, src_indata)

    def test_no_mapping_fields(self):
        src_indata = [
            {"id": "1234", "text": "Happy"},
            {"id": "1235", "text": "Gazonk"},
        ]
        align_it = AlignFieldNames(iter(src_indata), {})
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
        align_it = AlignFieldNames(iter(src_indata), {"id": "warcid", "text": "text"})
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
        align_it = AlignFieldNames(iter(src_indata), {"id": "id", "text": "context"})
        align_list = list(align_it)
        self.assertEqual(align_list, expected)


if __name__ == '__main__':
    unittest.main()
