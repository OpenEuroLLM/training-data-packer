import gzip
import shutil
import tempfile
import unittest
from pathlib import Path

import orjson
import zstandard as zstd
from parameterized import parameterized

from training_data_packer.utils import file


def _create_test_jsonl(path: Path, data: list) -> None:
    """Create a JSONL file for testing."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(orjson.dumps(item).decode("utf-8") + "\n")


def _create_test_gzip_jsonl(path: Path, data: list) -> None:
    """Create a gzipped JSONL file for testing."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for item in data:
            f.write(orjson.dumps(item).decode("utf-8") + "\n")


def _create_test_zst_jsonl(path: Path, data: list) -> None:
    """Create a zstandard-compressed JSONL file for testing."""
    cctx = zstd.ZstdCompressor()
    with open(path, "wb") as f:
        with cctx.stream_writer(f) as compressor:
            for item in data:
                compressor.write(orjson.dumps(item))
                compressor.write(b"\n")


class TestGenericJsonlReader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_read_uncompressed_jsonl(self):
        """Test reading from an uncompressed JSONL file."""
        test_data = [{"key": "value1"}, {"key": "value2"}, {"key": "value3"}]
        file_path = self.temp_path / "test.jsonl"
        _create_test_jsonl(file_path, test_data)

        reader = file.GenericJsonlReader(file_path)
        result = list(reader.read())

        self.assertEqual(result, test_data)
        self.assertEqual(reader._lines, 3)

    def test_read_gzip_jsonl(self):
        """Test reading from a gzipped JSONL file."""
        test_data = [{"key": "value1"}, {"key": "value2"}, {"key": "value3"}]
        file_path = self.temp_path / "test.jsonl.gz"
        _create_test_gzip_jsonl(file_path, test_data)

        reader = file.GenericJsonlReader(file_path)
        result = list(reader.read())

        self.assertEqual(result, test_data)
        self.assertEqual(reader._lines, 3)

    def test_read_zst_jsonl(self):
        """Test reading from a zstandard compressed JSONL file (.zst extension)."""
        test_data = [{"key": "value1"}, {"key": "value2"}, {"key": "value3"}]
        file_path = self.temp_path / "test.jsonl.zst"
        _create_test_zst_jsonl(file_path, test_data)

        reader = file.GenericJsonlReader(file_path)
        result = list(reader.read())

        self.assertEqual(result, test_data)
        self.assertEqual(reader._lines, 3)

    def test_read_zstd_jsonl(self):
        """Test reading from a zstandard compressed JSONL file (.zstd extension)."""
        test_data = [{"key": "value1"}, {"key": "value2"}, {"key": "value3"}]
        file_path = self.temp_path / "test.jsonl.zstd"
        _create_test_zst_jsonl(file_path, test_data)

        reader = file.GenericJsonlReader(file_path)
        result = list(reader.read())

        self.assertEqual(result, test_data)
        self.assertEqual(reader._lines, 3)

    def test_read_nonexistent_file(self):
        """Test reading from a file that doesn't exist."""
        file_path = self.temp_path / "nonexistent.jsonl"

        reader = file.GenericJsonlReader(file_path)
        result = list(reader.read())

        self.assertEqual(result, [])
        self.assertEqual(reader._lines, 0)

    def test_read_none_filename(self):
        """Test reading with None filename."""
        reader = file.GenericJsonlReader(None)
        result = list(reader.read())

        self.assertEqual(result, [])
        self.assertEqual(reader._lines, 0)

    def test_read_empty_string_filename(self):
        """Test reading with empty string filename."""
        reader = file.GenericJsonlReader("")
        result = list(reader.read())

        self.assertEqual(result, [])
        self.assertEqual(reader._lines, 0)

    def test_read_empty_file(self):
        """Test reading an empty JSONL file."""
        file_path = self.temp_path / "empty.jsonl"
        file_path.touch()

        reader = file.GenericJsonlReader(file_path)
        result = list(reader.read())

        self.assertEqual(result, [])
        self.assertEqual(reader._lines, 0)

    def test_read_different_encodings(self):
        """Test reading zstandard compressed files with different encodings."""
        test_data = [{"text": "Café"}, {"text": "naïve"}, {"text": "Über"}]
        file_path = self.temp_path / "test_cp1252.jsonl.zst"

        cctx = zstd.ZstdCompressor()
        with open(file_path, "wb") as f:
            with cctx.stream_writer(f) as compressor:
                for item in test_data:
                    json_str = orjson.dumps(item).decode("utf-8") + "\n"
                    compressor.write(json_str.encode("cp1252"))

        reader_cp1252 = file.GenericJsonlReader(file_path, encoding="cp1252")
        result_cp1252 = list(reader_cp1252.read())
        self.assertEqual(result_cp1252, test_data)

    @parameterized.expand(
        [
            ("nested_objects", [{"a": {"b": {"c": "deep"}}}, {"x": {"y": {"z": "nested"}}}]),
            ("arrays", [{"list": [1, 2, 3]}, {"list": ["a", "b", "c"]}]),
            ("mixed", [{"key": "value", "num": 42, "list": [1, 2, 3]}, {"nested": {"a": 1}}]),
            ("primitive_strings", [{"field": "value1"}, {"field": "value2"}]),
            ("primitive_numbers", [{"value": 123}, {"value": 456.789}]),
            ("primitive_booleans", [{"active": True}, {"active": False}]),
            ("primitive_nulls", [{"value": None}, {"value": "not null"}]),
        ]
    )
    def test_read_various_json_types(self, name, test_data):
        """Test reading JSONL files with various JSON data types."""
        file_path = self.temp_path / f"test_{name}.jsonl"
        _create_test_jsonl(file_path, test_data)

        reader = file.GenericJsonlReader(file_path)
        result = list(reader.read())

        self.assertEqual(result, test_data)

    def test_custom_counter_name(self):
        """Test with custom counter name in metrics."""
        test_data = [{"key": "value1"}, {"key": "value2"}]
        file_path = self.temp_path / "test.jsonl"
        _create_test_jsonl(file_path, test_data)

        reader = file.GenericJsonlReader(file_path, counter_name="custom_counter")
        list(reader.read())
        metrics = reader.get_metrics()

        self.assertIn("custom_counter", metrics)
        self.assertEqual(metrics["custom_counter"]["lines_read"], 2)

    def test_get_metrics(self):
        """Test getting metrics after reading."""
        test_data = [{"key": str(i)} for i in range(5)]
        file_path = self.temp_path / "test.jsonl"
        _create_test_jsonl(file_path, test_data)

        reader = file.GenericJsonlReader(file_path, counter_name="test_input")
        list(reader.read())
        metrics = reader.get_metrics()

        expected_metrics = {"test_input": {"lines_read": 5}}
        self.assertEqual(metrics, expected_metrics)

    def test_get_metrics_default_counter_name(self):
        """Test getting metrics with default counter name."""
        test_data = [{"key": "value"}]
        file_path = self.temp_path / "test.jsonl"
        _create_test_jsonl(file_path, test_data)

        reader = file.GenericJsonlReader(file_path)
        list(reader.read())
        metrics = reader.get_metrics()

        self.assertIn("input", metrics)
        self.assertEqual(metrics["input"]["lines_read"], 1)

    def test_get_metrics_no_lines_read(self):
        """Test getting metrics when no lines were read."""
        file_path = self.temp_path / "nonexistent.jsonl"

        reader = file.GenericJsonlReader(file_path)
        list(reader.read())
        metrics = reader.get_metrics()

        self.assertEqual(metrics["input"]["lines_read"], 0)

    def test_read_as_generator(self):
        """Test that read returns a generator and can be consumed multiple times."""
        test_data = [{"key": "value1"}, {"key": "value2"}]
        file_path = self.temp_path / "test.jsonl"
        _create_test_jsonl(file_path, test_data)

        reader = file.GenericJsonlReader(file_path)

        gen = reader.read()
        first_result = list(gen)

        gen2 = reader.read()
        second_result = list(gen2)

        self.assertEqual(first_result, test_data)
        self.assertEqual(second_result, test_data)
        self.assertEqual(reader._lines, 4)

    def test_large_file(self):
        """Test reading a larger file with many lines."""
        test_data = [{"index": i, "value": f"test_{i}"} for i in range(1000)]
        file_path = self.temp_path / "large.jsonl"
        _create_test_jsonl(file_path, test_data)

        reader = file.GenericJsonlReader(file_path)
        result = list(reader.read())

        self.assertEqual(len(result), 1000)
        self.assertEqual(reader._lines, 1000)
        self.assertEqual(result[0]["index"], 0)
        self.assertEqual(result[999]["index"], 999)


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
        """When the output file exists, should return False for should_continue."""
        output_file = self.temp_path / "output.jsonl"
        output_file.touch()

        tmp_path, should_continue = file.prepare_output_file(output_file)

        self.assertEqual(tmp_path, self.temp_path / ".output.jsonl")
        self.assertFalse(should_continue)

    def test_temp_file_exists(self):
        """When a temp file exists, should clean it up and return True for should_continue."""
        output_file = self.temp_path / "output.jsonl"
        temp_file = self.temp_path / ".output.jsonl"
        temp_file.touch()

        tmp_path, should_continue = file.prepare_output_file(output_file)

        self.assertEqual(tmp_path, self.temp_path / ".output.jsonl")
        self.assertTrue(should_continue)
        self.assertFalse(temp_file.exists())

    def test_neither_file_exists(self):
        """When neither file exists, should create directory and return True for should_continue."""
        output_file = self.temp_path / "subdir" / "output.jsonl"

        tmp_path, should_continue = file.prepare_output_file(output_file)

        self.assertEqual(tmp_path, self.temp_path / "subdir" / ".output.jsonl")
        self.assertTrue(should_continue)
        self.assertTrue(output_file.parent.exists())
        self.assertFalse(tmp_path.exists())


if __name__ == "__main__":
    unittest.main()
