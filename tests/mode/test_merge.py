import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import orjson as json
import zstandard as zstd

from training_data_packer.mode.merge import merge, process
from training_data_packer.utils.file import GenericJsonlReader


def _get_rows_in_file(file_path: Path) -> list[dict]:
    return list(GenericJsonlReader(file_path).read())


def _count_lines_in_file(file_path: Path) -> int:
    return len(_get_rows_in_file(file_path))


def _write_input(file_path: Path, ids: range):
    lines = b"".join(json.dumps({"id": i}) + b"\n" for i in ids)
    file_path.write_bytes(zstd.ZstdCompressor().compress(lines))


def _read_shards(destination_dir: Path) -> list[list[int]]:
    return [[row["id"] for row in _get_rows_in_file(p)] for p in sorted(destination_dir.glob("shard_*.jsonl.zst"))]


class MergeRestartTests(unittest.TestCase):
    """Three input files of five documents each, merged into shards of four documents."""

    _EXPECTED: ClassVar[list[list[int]]] = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14]]

    def setUp(self):
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        tmp = Path(self._tmpdir.name)
        self.input_files = [tmp.joinpath(f"in_{i}.jsonl.zst") for i in range(3)]
        for i, file_path in enumerate(self.input_files):
            _write_input(file_path, range(i * 5, i * 5 + 5))
        self.destination_dir = tmp.joinpath("release")

    def _state_path(self, shard: int) -> Path:
        return self.destination_dir.joinpath(f".shard_{shard:04d}.jsonl.zst.merge-state.json")

    def _state_files(self) -> list[str]:
        return sorted(p.name for p in self.destination_dir.glob(".*.merge-state.json"))

    def test_merge_removes_state_when_done(self):
        merge(self.input_files, self.destination_dir, 4, "shard")

        self.assertEqual(_read_shards(self.destination_dir), self._EXPECTED)
        self.assertEqual(self._state_files(), [])

    def test_merge_restarts_after_failure(self):
        self.input_files[2].write_bytes(b"not zstd")

        with self.assertRaises(zstd.ZstdError):
            merge(self.input_files, self.destination_dir, 4, "shard")

        self.assertEqual(_read_shards(self.destination_dir), self._EXPECTED[:2])
        self.assertEqual(self._state_files(), [".shard_0002.jsonl.zst.merge-state.json"])
        self.assertEqual(
            json.loads(self._state_path(2).read_bytes()),
            {
                "input_index": 1,
                "input_file": str(self.input_files[1]),
                "line": 3,
                "docs_per_shard": 4,
            },
        )
        self.assertTrue(self.destination_dir.joinpath(".shard_0002.jsonl.zst").exists())

        _write_input(self.input_files[2], range(10, 15))
        merge(self.input_files, self.destination_dir, 4, "shard")

        self.assertEqual(_read_shards(self.destination_dir), self._EXPECTED)
        self.assertEqual(list(self.destination_dir.glob(".*")), [])

    def test_merge_redoes_shard_renamed_before_state_was_saved(self):
        merge(self.input_files, self.destination_dir, 4, "shard")
        # Simulate a crash after shard 1 was renamed but before the state of shard 2 was saved
        for name in ["shard_0002.jsonl.zst", "shard_0003.jsonl.zst"]:
            self.destination_dir.joinpath(name).unlink()
        self._state_path(1).write_bytes(
            json.dumps(
                {
                    "input_index": 0,
                    "input_file": str(self.input_files[0]),
                    "line": 4,
                    "docs_per_shard": 4,
                },
            ),
        )

        merge(self.input_files, self.destination_dir, 4, "shard")

        self.assertEqual(_read_shards(self.destination_dir), self._EXPECTED)
        self.assertEqual(self._state_files(), [])

    def test_merge_restarts_from_latest_state(self):
        self.input_files[2].write_bytes(b"not zstd")
        with self.assertRaises(zstd.ZstdError):
            merge(self.input_files, self.destination_dir, 4, "shard")
        # Simulate a crash after the state of shard 2 was saved but before the state of shard 1 was removed
        state_of_shard_1 = {
            "input_index": 0,
            "input_file": str(self.input_files[0]),
            "line": 4,
            "docs_per_shard": 4,
        }
        self._state_path(1).write_bytes(json.dumps(state_of_shard_1))

        _write_input(self.input_files[2], range(10, 15))
        merge(self.input_files, self.destination_dir, 4, "shard")

        self.assertEqual(_read_shards(self.destination_dir), self._EXPECTED)
        self.assertEqual(self._state_files(), [])

    def test_merge_restart_rejects_changed_shard_size(self):
        self.input_files[2].write_bytes(b"not zstd")
        with self.assertRaises(zstd.ZstdError):
            merge(self.input_files, self.destination_dir, 4, "shard")

        with self.assertRaises(ValueError):
            merge(self.input_files, self.destination_dir, 3, "shard")

    def test_merge_restart_rejects_changed_input_files(self):
        self.input_files[2].write_bytes(b"not zstd")
        with self.assertRaises(zstd.ZstdError):
            merge(self.input_files, self.destination_dir, 4, "shard")

        with self.assertRaises(ValueError):
            merge(self.input_files[1:], self.destination_dir, 4, "shard")


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

    def test_merge_removes_unfinished_files(self):
        test_data = Path("tests/resources/integration/merge_flat_with_part")
        with TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir).joinpath("workdir")
            shutil.copytree(test_data, workdir)
            release_dir = workdir.joinpath("release")
            release_dir.mkdir()
            unfinished = release_dir.joinpath(".01_0_0007.jsonl.zst")
            unfinished.write_bytes(b"partial")
            other_hidden = release_dir.joinpath(".01_0_0000.jsonl.zst.metrics.json")
            other_hidden.write_text("{}")

            process(workdir)

            self.assertFalse(unfinished.exists())
            self.assertTrue(other_hidden.exists())
            self.assertEqual(_count_lines_in_file(release_dir.joinpath("01_0_0000.jsonl.zst")), 10)
            hidden_shards = [p.name for p in release_dir.glob(".*.jsonl.zst")]
            self.assertEqual(hidden_shards, [])


if __name__ == "__main__":
    unittest.main()
