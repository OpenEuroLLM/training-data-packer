import unittest
from pathlib import Path

from parameterized import parameterized

from training_data_packer.utils import file


class TestFileFunctions(unittest.TestCase):
    @parameterized.expand(
        [
            [
                "basic",
                "/foo/bar/hello.gz",
                ".gz",
                ".zst",
                "/foo/bar/hello.zst",
            ],
            [
                "mid_name",
                "/foo/bar/he.gz.llo.gz",
                ".gz",
                ".zst",
                "/foo/bar/he.gz.llo.zst",
            ],
            [
                "relative",
                "bar/hello.gz",
                ".gz",
                ".zst",
                "bar/hello.zst",
            ],
        ]
    )
    def test_something(self, name, filename, original_suffix, new_suffix, expected):
        new_name = file.change_suffix(filename, original_suffix, new_suffix)
        self.assertEqual(Path(expected), new_name)


if __name__ == "__main__":
    unittest.main()
