import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from training_data_packer.merge import process
from training_data_packer.utils.file import GenericJsonlReader


def _get_rows_in_file(file_path: Path) -> list[dict]:
    return list(GenericJsonlReader(file_path).read())


def _count_lines_in_file(file_path: Path) -> int:
    return len(_get_rows_in_file(file_path))


class MergeIntegrationTests(unittest.TestCase):
    def test_merge_flat_with_part(self):
        test_data = Path("tests/resources/integration/merge_flat_with_part")
        with TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir).joinpath("workdir")
            shutil.copytree(test_data, workdir)

            process(workdir)

            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/01_0_0000.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/01_0_0001.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/01_0_0002.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/01_0_0003.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/01_0_0004.jsonl.zst")), 6)

            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/02_0_0000.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/02_0_0001.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/02_0_0002.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/02_0_0003.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/02_0_0004.jsonl.zst")), 6)

    def test_merge_flat_without_part(self):
        test_data = Path("tests/resources/integration/merge_flat_without_part_prefix")
        with TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir).joinpath("workdir")
            shutil.copytree(test_data, workdir)

            process(workdir)

            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard_0000.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard_0001.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard_0002.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard_0003.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard_0004.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard_0005.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard_0006.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard_0007.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard_0008.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard_0009.jsonl.zst")), 2)

    def test_merge_tree(self):
        test_data = Path("tests/resources/integration/merge_tree_without_prefix")
        with TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir).joinpath("workdir")
            shutil.copytree(test_data, workdir)

            process(workdir)

            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard01/shard_0000.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard01/shard_0001.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard01/shard_0002.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard01/shard_0003.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard01/shard_0004.jsonl.zst")), 6)

            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard02/shard_0000.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard02/shard_0001.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard02/shard_0002.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard02/shard_0003.jsonl.zst")), 10)
            self.assertEqual(_count_lines_in_file(workdir.joinpath("release/shard02/shard_0004.jsonl.zst")), 6)


if __name__ == "__main__":
    unittest.main()
