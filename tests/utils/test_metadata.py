import unittest

from training_data_packer.utils.metadata import get_matching_part


class MyTestCase(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
