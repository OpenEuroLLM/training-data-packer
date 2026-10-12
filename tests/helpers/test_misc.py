import unittest

from parameterized import parameterized

from training_data_packer.utils.misc import get_dict_value, get_dict_values


class TestGetDictValue(unittest.TestCase):
    @parameterized.expand(
        [
            ("top_level", {"name": "example"}, "name", "example"),
            ("nested", {"metadata": {"language": "sv"}}, "metadata.language", "sv"),
            ("list_item", {"items": [{"value": 42}]}, "items[0].value", 42),
            ("matched_none", {"value": None}, "value", None),
        ]
    )
    def test_returns_single_match(self, _name, dictionary, key, expected):
        self.assertEqual(get_dict_value(dictionary, key, default="fallback"), expected)

    def test_returns_none_when_key_is_missing(self):
        self.assertIsNone(get_dict_value({"present": True}, "missing"))

    def test_returns_custom_default_when_key_is_missing(self):
        default = object()

        self.assertIs(get_dict_value({"present": True}, "missing", default), default)

    def test_raises_key_error_when_expression_has_multiple_matches(self):
        dictionary = {"items": [{"value": 1}, {"value": 2}]}

        with self.assertRaisesRegex(KeyError, r"items\[\*\]\.value gives multiple hits"):
            get_dict_value(dictionary, "items[*].value")


class TestGetDictValues(unittest.TestCase):
    @parameterized.expand(
        [
            ("single_match", {"name": "example"}, "name", ["example"]),
            (
                "multiple_matches",
                {"items": [{"value": 1}, {"value": 2}, {"value": 3}]},
                "items[*].value",
                [1, 2, 3],
            ),
            (
                "nested_matches",
                {"groups": {"a": {"value": 1}, "b": {"value": 2}}},
                "groups.*.value",
                [1, 2],
            ),
            ("matched_none", {"items": [{"value": None}]}, "items[*].value", [None]),
        ]
    )
    def test_returns_all_matches(self, _name, dictionary, key, expected):
        self.assertEqual(get_dict_values(dictionary, key), expected)

    def test_returns_empty_list_when_expression_has_no_matches(self):
        self.assertEqual(get_dict_values({"items": []}, "items[*].value"), [])


if __name__ == "__main__":
    unittest.main()
