import shutil
import tempfile
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
                "same",
                "/foo/bar/hello.gz",
                ".gz",
                ".gz",
                "/foo/bar/hello.gz",
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


class TestPrepareOutputFile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_output_file_exists(self):
        """When the output file exists, should return False for should_continue"""
        output_file = self.temp_path / "output.jsonl"
        output_file.touch()

        tmp_path, should_continue = file.prepare_output_file(output_file)

        self.assertEqual(tmp_path, self.temp_path / ".output.jsonl")
        self.assertFalse(should_continue)

    def test_temp_file_exists(self):
        """When a temp file exists, should clean it up and return True for should_continue"""
        output_file = self.temp_path / "output.jsonl"
        temp_file = self.temp_path / ".output.jsonl"
        temp_file.touch()

        tmp_path, should_continue = file.prepare_output_file(output_file)

        self.assertEqual(tmp_path, self.temp_path / ".output.jsonl")
        self.assertTrue(should_continue)
        self.assertFalse(temp_file.exists())

    def test_neither_file_exists(self):
        """When neither file exists, should create directory and return True for should_continue"""
        output_file = self.temp_path / "subdir" / "output.jsonl"

        tmp_path, should_continue = file.prepare_output_file(output_file)

        self.assertEqual(tmp_path, self.temp_path / "subdir" / ".output.jsonl")
        self.assertTrue(should_continue)
        self.assertTrue(output_file.parent.exists())
        self.assertFalse(tmp_path.exists())


if __name__ == "__main__":
    unittest.main()
