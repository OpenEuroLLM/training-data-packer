import unittest

from training_data_packer.utils.iterator import get_until_key_change


class TestIteratorGetUntilKeyChange(unittest.TestCase):
    def test_happy_path(self):
        indata = ["A", "B", "AB"]
        key, values, iterator = get_until_key_change(iter(indata), len)
        self.assertEqual(key, 1)
        self.assertEqual(values, ["A", "B"])
        key, values, iterator = get_until_key_change(iterator, len)
        self.assertEqual(key, 2)
        self.assertEqual(values, ["AB"])

    def test_call_empty_iterator(self):
        with self.assertRaises(StopIteration):
            key, values, iterator = get_until_key_change(iter([]), len)


if __name__ == '__main__':
    unittest.main()
