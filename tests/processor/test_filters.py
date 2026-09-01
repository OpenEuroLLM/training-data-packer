import unittest

from training_data_packer.processor.filters import FilterOnBlocklist


class TestFilterOnBlocklist(unittest.TestCase):
    def test_filtering_with_metrics(self):
        blocklist = {"2", "4", "7"}
        input_elements = [
            {"id": "1", "name": "block1"},
            {"id": "2", "name": "block1"},
            {"id": "3", "name": "block1"},
            {"id": "4", "name": "block1"},
            {"id": "5", "name": "block1"},
            {"id": "6", "name": "block1"},
        ]
        expected_elements = [
            {"id": "1", "name": "block1"},
            {"id": "3", "name": "block1"},
            {"id": "5", "name": "block1"},
            {"id": "6", "name": "block1"},
        ]
        to_test = FilterOnBlocklist("test_filter", blocklist)
        result = list(to_test.filter(input_elements))
        self.assertEqual(result, expected_elements)
        self.assertEqual(
            to_test.get_metrics(),
            {
                "test_filter": {
                    "removed": 2,
                    "list_length": 3,
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
