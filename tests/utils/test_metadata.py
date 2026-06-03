import unittest

from parameterized import parameterized

from training_data_packer.utils.metadata import get_matching_part, get_metadata_value, get_shard_size_documents


class TestMetadata(unittest.TestCase):
    def test_metadata_without_defaults(self):
        indata = {"release": {"foo": {"sample": "full"}, "bar": {"sample": "full"}}}
        self.assertEqual(get_matching_part(indata, "bla/foo/shard01"), ({"sample": "full"}, "foo"))

    def test_metadata_merge_defaults(self):
        indata = {"release": {"default": {"pack": "tree"}, "foo": {"sample": "full"}, "bar": {"sample": "full"}}}
        self.assertEqual(get_matching_part(indata, "bla/foo/shard01"), ({"sample": "full", "pack": "tree"}, "foo"))

    def test_metadata_override_defaults(self):
        indata = {
            "release": {
                "default": {
                    "pack": "tree",
                    "sample": "wds",
                },
                "foo": {
                    "sample": "full",
                },
                "bar": {
                    "sample": "full",
                },
            }
        }
        self.assertEqual(get_matching_part(indata, "bla/foo/shard01"), ({"sample": "full", "pack": "tree"}, "foo"))

    def test_no_matching_part(self):
        indata = {
            "release": {
                "default": {
                    "pack": "tree",
                    "sample": "wds",
                },
                "bar": {
                    "sample": "full",
                },
            }
        }
        with self.assertRaises(ValueError):
            get_matching_part(indata, "bla/foo/shard01")

    def test_get_shard_size_documents(self):
        self.assertEqual(10_000_000_000, get_shard_size_documents({"shard": "10bd"}))
        self.assertEqual(5_000_000, get_shard_size_documents({"shard": "5md"}))
        self.assertEqual(350, get_shard_size_documents({"shard": "350"}))
        with self.assertRaises(ValueError):
            get_shard_size_documents({"shard": "350tt"})

    @parameterized.expand(
        [
            ["Value_set", {"foo": "bar"}, "foo", "gazsonk", "bar"],
            ["default", {"foo": "bar"}, "key", "gazonk", "gazonk"],
            [
                "hierarchy",
                {"foo": {"bar": 42}},
                "foo.bar",
                17,
                42,
            ],
        ]
    )
    def test_get_metadata_value(self, name, metadata, key, default_value, expected):
        result = get_metadata_value(metadata, key, default_value)
        self.assertEqual(expected, result)


if __name__ == "__main__":
    unittest.main()
