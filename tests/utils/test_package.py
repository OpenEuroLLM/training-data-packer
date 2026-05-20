import unittest
from pathlib import Path

from training_data_packer import package


class TestCalculateFilePath(unittest.TestCase):
    src_dir = Path("/srv/data/gazonk/source")
    contamination_dir = Path("/srv/data/gazonk/contamination")
    pii_dir = Path("/srv/data/gazonk/pii")
    output_dir = Path("/srv/data/gazonk/release_raw")

    def test_calculate_file_paths_default(self):
        src_file = self.src_dir.joinpath("shard01/file01.jsonl.zst")
        metadata = {"suffix": ".jsonl.zstd"}
        contamination_file, pii_file, out_file = package._calculate_file_paths(
            src_file, self.src_dir, self.contamination_dir, self.pii_dir, self.output_dir, metadata
        )
        self.assertEqual(self.contamination_dir.joinpath("shard01/file01.jsonl.zst"), contamination_file)
        self.assertEqual(self.pii_dir.joinpath("shard01/file01.jsonl.zst"), pii_file)
        self.assertEqual(self.output_dir.joinpath("shard01/file01.jsonl.zst"), out_file)

    def test_calculate_file_paths_contamination_suffix(self):
        src_file = self.src_dir.joinpath("shard01/file01.jsonl.zst")
        metadata = {"suffix": ".jsonl.zst", "annotations": {"contamination": {"suffix": ".jsonl"}}}
        contamination_file, pii_file, out_file = package._calculate_file_paths(
            src_file, self.src_dir, self.contamination_dir, self.pii_dir, self.output_dir, metadata
        )
        self.assertEqual(self.contamination_dir.joinpath("shard01/file01.jsonl"), contamination_file)
        self.assertEqual(self.pii_dir.joinpath("shard01/file01.jsonl.zst"), pii_file)
        self.assertEqual(self.output_dir.joinpath("shard01/file01.jsonl.zst"), out_file)

    def test_calculate_file_paths_pii_suffix(self):
        src_file = self.src_dir.joinpath("shard01/file01.jsonl.zst")
        metadata = {"suffix": ".jsonl.zst", "annotations": {"pii": {"suffix": ".jsonl"}}}
        contamination_file, pii_file, out_file = package._calculate_file_paths(
            src_file, self.src_dir, self.contamination_dir, self.pii_dir, self.output_dir, metadata
        )
        self.assertEqual(self.contamination_dir.joinpath("shard01/file01.jsonl.zst"), contamination_file)
        self.assertEqual(self.pii_dir.joinpath("shard01/file01.jsonl"), pii_file)
        self.assertEqual(self.output_dir.joinpath("shard01/file01.jsonl.zst"), out_file)

    def test_calculate_file_paths_other_suffix(self):
        src_file = self.src_dir.joinpath("shard01/file01.jsonl.gz")
        metadata = {"suffix": ".jsonl.gz"}
        contamination_file, pii_file, out_file = package._calculate_file_paths(
            src_file, self.src_dir, self.contamination_dir, self.pii_dir, self.output_dir, metadata
        )
        self.assertEqual(self.contamination_dir.joinpath("shard01/file01.jsonl.gz"), contamination_file)
        self.assertEqual(self.pii_dir.joinpath("shard01/file01.jsonl.gz"), pii_file)
        self.assertEqual(self.output_dir.joinpath("shard01/file01.jsonl.zst"), out_file)


if __name__ == "__main__":
    unittest.main()
