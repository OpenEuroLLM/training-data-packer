import unittest
from pathlib import Path

from parameterized import parameterized

from training_data_packer import propella_structure


class TestComputeOutputFilename(unittest.TestCase):
    @parameterized.expand(
        [
            [
                "basic",
                "/foo/bar/propella-4b",
                "shard01/000.jsonl.zst",
                {"suffix": ".jsonl.zst"},
                "/foo/bar/propella-4b/shard01/000.jsonl.zst",
            ],
            [
                "zstd",
                "/foo/bar/propella-4b",
                "shard01/000.jsonl.zstd",
                {"suffix": ".jsonl.zstd", "annotations": {"propella-4b": {"suffix": ".jsonl.zst"}}},
                "/foo/bar/propella-4b/shard01/000.jsonl.zst",
            ],
            [
                "no_propella_suffix_not_zst",
                "/foo/bar/propella-4b",
                "shard01/000.jsonl.gz",
                {"suffix": ".jsonl.gz"},
                "/foo/bar/propella-4b/shard01/000.jsonl.gz",
            ],
        ]
    )
    def test_something(self, name, output_dir, rel_path, metadata, expected):
        result = propella_structure._compute_output_filename(Path(output_dir), Path(rel_path), metadata)
        self.assertEqual(Path(expected), result)


if __name__ == "__main__":
    unittest.main()
