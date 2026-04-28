import unittest

from training_data_packer.decontaminate import Decontaminate


class DecontaminationTest(unittest.TestCase):
    def test_happy_path(self):
        src_indata=[
            {"id": "1234"},
            {"id": "1235"},
            {"id": "1236"},
        ]
        decont_indata=[
            {"id": "1235"},
        ]
        dit = Decontaminate(iter(src_indata), iter(decont_indata))
        l=list(dit)
        self.assertEqual(
            l,
            [
                {'delete': False, 'id': '1234'},
                {'delete': True, 'id': '1235'},
                {'delete': False, 'id': '1236'}
            ])

    def test_there_are_contamination_records_left(self):
        src_indata=[
            {"id": "1234"},
        ]
        decont_indata=[
            {"id": "1237"},
        ]
        dit = Decontaminate(iter(src_indata), iter(decont_indata))
        with self.assertRaises(ValueError):
            list(dit)

    def test_test_all_objects_to_remove(self):
        src_indata=[
            {"id": "1234"},
            {"id": "1235"},
            {"id": "1236"},
        ]
        decont_indata=[
            {"id": "1234"},
            {"id": "1235"},
            {"id": "1236"},
        ]
        dit = Decontaminate(iter(src_indata), iter(decont_indata))
        l=list(dit)
        self.assertEqual(
            l,
            [
                {'delete': True, 'id': '1234'},
                {'delete': True, 'id': '1235'},
                {'delete': True, 'id': '1236'}
            ])

    def test_no_contamination(self):
        src_indata=[
            {"id": "1234"},
            {"id": "1235"},
        ]
        decont_indata=[]
        dit = Decontaminate(iter(src_indata), iter(decont_indata))
        l=list(dit)
        self.assertEqual(
            l,
            [
                {'delete': False, 'id': '1234'},
                {'delete': False, 'id': '1235'},
            ])


if __name__ == '__main__':
    unittest.main()
