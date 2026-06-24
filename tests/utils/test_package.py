import unittest
from pathlib import Path

from training_data_packer import package


class TestCalculateFilePath(unittest.TestCase):
    src_dir = Path("/srv/data/gazonk/source")
    contamination_dir = Path("/srv/data/gazonk/contamination")
    pii_dir = Path("/srv/data/gazonk/pii")
    propella_dir = Path("/srv/data/gazonk/propella-4b")
    output_dir = Path("/srv/data/gazonk/release_raw")

    def test_calculate_file_paths_default(self):
        src_file = self.src_dir.joinpath("shard01/file01.jsonl.zst")
        metadata = {
            "_internal": {"collection_dir": "/srv/data/gazonk"},
            "release": {"default": {"input": "source"}},
            "suffix": ".jsonl.zstd",
        }
        contamination_file = package._calculate_file_path(src_file, metadata, "release", self.contamination_dir)
        pii_file = package._calculate_file_path(src_file, metadata, "release", self.pii_dir)
        propella_file = package._calculate_file_path(src_file, metadata, "release", self.propella_dir)
        out_file = package._calculate_file_path(src_file, metadata, "release", self.output_dir)

        self.assertEqual(self.contamination_dir.joinpath("shard01/file01.jsonl.zst"), contamination_file)
        self.assertEqual(self.pii_dir.joinpath("shard01/file01.jsonl.zst"), pii_file)
        self.assertEqual(self.propella_dir.joinpath("shard01/file01.jsonl.zst"), propella_file)
        self.assertEqual(self.output_dir.joinpath("shard01/file01.jsonl.zst"), out_file)

    def test_calculate_file_paths_source_suffix(self):
        src_file = self.src_dir.joinpath("shard01/file01.jsonl.gz")
        metadata = {
            "_internal": {"collection_dir": "/srv/data/gazonk"},
            "release": {"default": {"input": "source"}},
            "source": {"default": {"suffix": ".jsonl.gz"}},
            "suffix": ".jsonl.zst",
        }

        contamination_file = package._calculate_file_path(src_file, metadata, "release", self.contamination_dir)
        pii_file = package._calculate_file_path(src_file, metadata, "release", self.pii_dir)
        propella_file = package._calculate_file_path(src_file, metadata, "release", self.propella_dir)
        out_file = package._calculate_file_path(src_file, metadata, "release", self.output_dir)

        self.assertEqual(self.contamination_dir.joinpath("shard01/file01.jsonl.zst"), contamination_file)
        self.assertEqual(self.pii_dir.joinpath("shard01/file01.jsonl.zst"), pii_file)
        self.assertEqual(self.propella_dir.joinpath("shard01/file01.jsonl.zst"), propella_file)
        self.assertEqual(self.output_dir.joinpath("shard01/file01.jsonl.zst"), out_file)

    def test_calculate_file_paths_sample_source_suffix(self):
        src_file = self.src_dir.joinpath("shard01/file01.jsonl.gz")
        output_dir = Path("/srv/data/gazonk/sample")
        metadata = {
            "_internal": {"collection_dir": "/srv/data/gazonk"},
            "sample": {"default": {"input": "source"}},
            "source": {"default": {"suffix": ".jsonl.gz"}},
            "suffix": ".jsonl.zst",
        }

        contamination_file = package._calculate_file_path(src_file, metadata, "sample", self.contamination_dir)
        pii_file = package._calculate_file_path(src_file, metadata, "sample", self.pii_dir)
        propella_file = package._calculate_file_path(src_file, metadata, "sample", self.propella_dir)
        out_file = package._calculate_file_path(src_file, metadata, "sample", output_dir)

        self.assertEqual(self.contamination_dir.joinpath("shard01/file01.jsonl.zst"), contamination_file)
        self.assertEqual(self.pii_dir.joinpath("shard01/file01.jsonl.zst"), pii_file)
        self.assertEqual(self.propella_dir.joinpath("shard01/file01.jsonl.zst"), propella_file)
        self.assertEqual(output_dir.joinpath("shard01/file01.jsonl.zst"), out_file)


if __name__ == "__main__":
    unittest.main()
