import unittest

from parameterized import parameterized

from training_data_packer.utils.slurm import _get_my_partition_tasks


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
