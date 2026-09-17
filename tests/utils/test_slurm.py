import os
import tempfile
import unittest
from pathlib import Path

from parameterized import parameterized

from training_data_packer.metadata import Metadata
from training_data_packer.utils.slurm import _get_my_partition_tasks, schedule_files


def _mock_function(file_path: Path, metadata: Metadata):
    pass


def _failing_function(file_path: Path, metadata: Metadata):
    raise ValueError(f"Processing failed for {file_path}")


class TestAlignFieldNames(unittest.TestCase):
    @parameterized.expand(
        [
            ["one_task", ["a", "b"], 1, 1, ["a", "b"]],
            ["two_tasks_id_one", ["a", "b"], 2, 1, ["a"]],
            ["two_tasks_id_two", ["a", "b"], 2, 2, ["b"]],
            ["two_tasks_id_one_with_rest", ["a", "b", "c"], 2, 1, ["a", "b"]],
            ["two_tasks_id_two_with_rest", ["a", "b", "c"], 2, 2, ["c"]],
            [
                "larger_example_1",
                ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
                3,
                1,
                ["a", "b", "c", "d"],
            ],
            [
                "larger_example_2",
                ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
                3,
                2,
                ["e", "f", "g", "h"],
            ],
            [
                "larger_example_3",
                ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
                3,
                3,
                ["i", "j", "k"],
            ],
            ["fewer_files_than_tasks_1", ["a"], 2, 1, ["a"]],
            ["fewer_files_than_tasks_2", ["a"], 2, 2, []],
        ]
    )
    def test_get_my_partition_tasks(self, name, files, task_count, task_id, expected_files):
        result = _get_my_partition_tasks(files, task_count, task_id)
        self.assertEqual(result, expected_files)


class TestScheduleFiles(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_files = [Path(self.temp_dir.name) / f"file{i}.txt" for i in range(5)]
        for file_path in self.test_files:
            file_path.touch()
        self.metadata = Metadata({"test": "data"})

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_schedule_files_sequential(self):
        schedule_files(self.test_files, self.metadata, _mock_function, workers=1, slurm=False)

    def test_schedule_files_parallel(self):
        schedule_files(self.test_files, self.metadata, _mock_function, workers=2, slurm=False)

    def test_schedule_files_slurm_mode(self):
        os.environ["SLURM_ARRAY_TASK_COUNT"] = "2"
        os.environ["SLURM_ARRAY_TASK_ID"] = "1"
        try:
            schedule_files(self.test_files, self.metadata, _mock_function, workers=1, slurm=True)
        finally:
            del os.environ["SLURM_ARRAY_TASK_COUNT"]
            del os.environ["SLURM_ARRAY_TASK_ID"]

    def test_schedule_files_parallel_error_handling(self):
        with self.assertRaises(RuntimeError) as context:
            schedule_files(self.test_files, self.metadata, _failing_function, workers=2, slurm=False)
        self.assertEqual(str(context.exception), "One or more workers failed")

    def test_schedule_files_empty_file_list(self):
        schedule_files([], self.metadata, _mock_function, workers=1, slurm=False)
