import os
import tempfile
import unittest
from pathlib import Path

import yaml
from parameterized import parameterized

from training_data_packer.mode.plan import GIB, StepPlan, plan_step, process


class TestStepPlan(unittest.TestCase):
    @parameterized.expand(
        [
            ["rounds_up_to_hours", 61, "--partition=small --array=0-3 --time=2:00:00"],
            ["at_least_one_hour", 0.5, "--partition=small --array=0-3 --time=1:00:00"],
            ["whole_hours", 300, "--partition=small --array=0-3 --time=5:00:00"],
        ]
    )
    def test_slurm_args(self, name, minutes, expected):
        self.assertEqual(StepPlan("package", "small", 4, minutes).slurm_args(), expected)


class TestPlanStep(unittest.TestCase):
    @parameterized.expand(
        [
            # 3000 GiB over 8 parallel files gives 2 tasks of 5 hours ideally, files of 10 GiB
            ["package_ideal", [10 * GIB] * 300, 8, 300, 2, 187.5],
            # Fewer files than parallel slots in a task, so each task gets one file
            ["package_few_files", [10 * GIB] * 4, 8, 4, 1, 10],
            # Largest file sets the minimum time
            ["largest_file", [600 * GIB, GIB], 8, 2, 1, 600],
            # One file at a time, 900 GiB gives 3 tasks
            ["sequential", [GIB] * 900, 1, 900, 3, 300],
            # Limited by max tasks, e.g. flat merge
            ["max_tasks", [GIB] * 900, 1, 1, 1, 900],
            # Can not have more tasks than units
            ["max_tasks_units", [400 * GIB, 400 * GIB], 1, 10, 2, 400],
        ]
    )
    def test_plan_step(self, name, sizes, parallel, max_tasks, expected_array_size, expected_minutes):
        result = plan_step("step", "small", sizes, parallel, max_tasks)
        self.assertEqual(result.array_size, expected_array_size)
        self.assertAlmostEqual(result.minutes, expected_minutes)

    @parameterized.expand(
        [
            # 1200 files would ideally be 1200 tasks of 5 hours
            ["small", "small", 300 * GIB, 200, 1800],
            ["standard", "standard", 240 * GIB, 100, 2880],
        ]
    )
    def test_partition_max_tasks(self, name, partition, file_size, expected_array_size, expected_minutes):
        result = plan_step("step", partition, [file_size] * 1200, 1, 1200)
        self.assertEqual(result.array_size, expected_array_size)
        self.assertAlmostEqual(result.minutes, expected_minutes)

    def test_safety_ratio(self):
        # 900 GiB with ratio 1.5 is 1350 minutes, 5 tasks of 270 minutes
        result = plan_step("step", "small", [GIB] * 900, 1, 900, safety_ratio=1.5)
        self.assertEqual(result.array_size, 5)
        self.assertAlmostEqual(result.minutes, 270)

    def test_safety_ratio_exceeds_partition(self):
        # 48 hours fits in partition standard, but not with the safety ratio
        with self.assertRaises(ValueError):
            plan_step("tokenize", "standard", [48 * 60 * GIB], 1, 1, safety_ratio=1.1)

    def test_exceeds_partition_max_tasks(self):
        # 101 files of 48 hours each do not fit in 100 tasks of 48 hours
        with self.assertRaises(ValueError):
            plan_step("tokenize", "standard", [48 * 60 * GIB] * 101, 1, 101)

    def test_exceeds_partition(self):
        with self.assertRaises(ValueError):
            plan_step("tokenize", "standard", [49 * 60 * GIB], 1, 1)

    def test_fits_partition(self):
        result = plan_step("merge", "small", [72 * 60 * GIB], 1, 1)
        self.assertEqual(result.slurm_args(), "--partition=small --array=0-0 --time=72:00:00")


def _create_sparse_file(path: Path, size: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        file.truncate(size)


class TestProcess(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.collection_dir = Path(self.temp_dir.name)
        for part in ["part01", "part02", "not-in-metadata"]:
            for n in range(10):
                _create_sparse_file(self.collection_dir / "source" / part / f"file_{n:02d}.jsonl.zst", 100 * GIB)
        # Hidden files are not included
        _create_sparse_file(self.collection_dir / "source" / "part01" / ".hidden.jsonl.zst", 1000 * GIB)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_metadata(self, pack: str, prefix: bool = False):
        part = {"prefix": "p"} if prefix else None
        metadata = {
            "name": "dataset name",
            "megatron-lm": {"default": {"source": "release"}, "dummy-tokenizer": {"huggingface": "dummy/tokenizer"}},
            "source": {"default": None, "part01": None, "part02": None},
            "release": {"default": {"input": "source", "pack": pack, "shard": "10d"}, "part01": part, "part02": part},
        }
        with open(self.collection_dir / "metadata.yaml", "w") as file:
            yaml.safe_dump(metadata, file)

    def test_tree(self):
        self._write_metadata("tree")

        package, merge, tokenize = process(self.collection_dir, safety_ratio=1.0)

        # 2000 GiB in files of 100 GiB, ideal 2400 GiB per task with 8 parallel files
        self.assertEqual(package.slurm_args(), "--partition=small --array=0-0 --time=5:00:00")
        # Two parts of 1000 GiB each
        self.assertEqual(merge.slurm_args(), "--partition=small --array=0-1 --time=17:00:00")
        # 2000 GiB, ideal 300 GiB per task
        self.assertEqual(tokenize.slurm_args(), "--partition=standard --array=0-6 --time=5:00:00")

    def test_tree_default_safety_ratio(self):
        self._write_metadata("tree")

        package, merge, tokenize = process(self.collection_dir)

        # 2000 GiB with ratio 2 is 3500 minutes, 2 tasks of 8 parallel files
        self.assertEqual(package.slurm_args(), "--partition=small --array=0-1 --time=5:00:00")
        # Two parts of 1750 minutes each
        self.assertEqual(merge.slurm_args(), "--partition=small --array=0-1 --time=34:00:00")
        # 3500 minutes, ideal 300 minutes per task
        self.assertEqual(tokenize.slurm_args(), "--partition=standard --array=0-13 --time=5:00:00")

    def test_invalid_safety_ratio(self):
        self._write_metadata("tree")

        with self.assertRaises(ValueError):
            process(self.collection_dir, safety_ratio=0)

    def test_flat(self):
        self._write_metadata("flat")

        _, merge, _ = process(self.collection_dir, safety_ratio=1.0)

        self.assertEqual(merge.slurm_args(), "--partition=small --array=0-0 --time=34:00:00")

    def test_flat_with_prefix(self):
        self._write_metadata("flat", prefix=True)

        _, merge, _ = process(self.collection_dir, safety_ratio=1.0)

        self.assertEqual(merge.slurm_args(), "--partition=small --array=0-1 --time=17:00:00")

    def test_no_files(self):
        self._write_metadata("tree")
        for file in (self.collection_dir / "source").glob("**/*.zst"):
            os.remove(file)

        with self.assertRaises(ValueError):
            process(self.collection_dir)


if __name__ == "__main__":
    unittest.main()
