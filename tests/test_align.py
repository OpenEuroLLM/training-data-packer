import unittest

from training_data_packer.align import AlignFieldNames


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
