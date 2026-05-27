import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from training_data_packer.storage.propella import query_field


def _create_parquet_file(path: Path, data: list[dict]) -> None:
    """Helper function to create a parquet file from data."""
    table_data: dict[str, list] = {}
    for row in data:
        for key, value in row.items():
            if key not in table_data:
                table_data[key] = []
            table_data[key].append(value)
    table = pa.table(table_data)
    pq.write_table(table, path)


class TestQueryField(unittest.TestCase):
    def test_query_field_with_matching_records(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            file1 = tmp_path / "file1.parquet"
            file2 = tmp_path / "file2.parquet"

            _create_parquet_file(
                file1,
                [
                    {"id": 1, "name": "Alice", "status": "active"},
                    {"id": 2, "name": "Bob", "status": "inactive"},
                ],
            )
            _create_parquet_file(
                file2,
                [
                    {"id": 3, "name": "Charlie", "status": "active"},
                    {"id": 4, "name": "David", "status": "inactive"},
                ],
            )

            result = query_field(tmp_path, "status", "active")
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["name"], "Alice")
            self.assertEqual(result[1]["name"], "Charlie")

    def test_query_field_with_no_matches(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            file1 = tmp_path / "file1.parquet"
            _create_parquet_file(
                file1,
                [
                    {"id": 1, "name": "Alice", "status": "active"},
                    {"id": 2, "name": "Bob", "status": "active"},
                ],
            )

            result = query_field(tmp_path, "status", "inactive")
            self.assertEqual(len(result), 0)

    def test_query_field_with_numeric_value(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            file1 = tmp_path / "file1.parquet"
            _create_parquet_file(
                file1,
                [
                    {"id": 1, "name": "Alice", "score": 85},
                    {"id": 2, "name": "Bob", "score": 90},
                    {"id": 3, "name": "Charlie", "score": 85},
                ],
            )

            result = query_field(tmp_path, "score", 85)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["name"], "Alice")
            self.assertEqual(result[1]["name"], "Charlie")

    def test_query_field_invalid_field_name_with_dash(self):
        with TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as cm:
                query_field(tmpdir, "field-name", "value")
            self.assertIn("only alphanumeric characters and underscores are allowed", str(cm.exception))

    def test_query_field_invalid_field_name_with_space(self):
        with TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as cm:
                query_field(tmpdir, "field name", "value")
            self.assertIn("only alphanumeric characters and underscores are allowed", str(cm.exception))

    def test_query_field_invalid_field_name_with_dot(self):
        with TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as cm:
                query_field(tmpdir, "field.name", "value")
            self.assertIn("only alphanumeric characters and underscores are allowed", str(cm.exception))

    def test_query_field_invalid_field_name_with_special_chars(self):
        with TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as cm:
                query_field(tmpdir, "field@name", "value")
            self.assertIn("only alphanumeric characters and underscores are allowed", str(cm.exception))

    def test_query_field_valid_field_name_alphanumeric_only(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            file1 = tmp_path / "file1.parquet"
            _create_parquet_file(
                file1,
                [
                    {"id": 1, "abc123": "value1"},
                    {"id": 2, "abc123": "value2"},
                ],
            )

            result = query_field(tmp_path, "abc123", "value1")
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["id"], 1)

    def test_query_field_valid_field_name_with_underscores(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            file1 = tmp_path / "file1.parquet"
            _create_parquet_file(
                file1,
                [
                    {"id": 1, "field_name": "value1"},
                    {"id": 2, "field_name": "value2"},
                ],
            )

            result = query_field(tmp_path, "field_name", "value1")
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["id"], 1)

    def test_query_field_not_a_directory(self):
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.parquet"
            test_file.write_text("not a directory")
            with self.assertRaises(NotADirectoryError):
                query_field(test_file, "valid_field", "value")

    def test_query_field_no_parquet_files(self):
        with TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError) as cm:
                query_field(tmpdir, "valid_field", "value")
            self.assertIn("No parquet files found", str(cm.exception))

    def test_query_field_with_single_file(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            file1 = tmp_path / "data.parquet"
            _create_parquet_file(
                file1,
                [
                    {"id": 1, "name": "Alice", "status": "active"},
                    {"id": 2, "name": "Bob", "status": "inactive"},
                    {"id": 3, "name": "Charlie", "status": "active"},
                ],
            )

            result = query_field(tmp_path, "status", "active")
            self.assertEqual(len(result), 2)
            names = [r["name"] for r in result]
            self.assertEqual(names, ["Alice", "Charlie"])

    def test_query_field_with_multiple_parquet_files(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _create_parquet_file(tmp_path / "part1.parquet", [{"id": 1, "key": "alpha"}, {"id": 2, "key": "beta"}])
            _create_parquet_file(tmp_path / "part2.parquet", [{"id": 3, "key": "alpha"}, {"id": 4, "key": "gamma"}])
            _create_parquet_file(tmp_path / "part3.parquet", [{"id": 5, "key": "alpha"}, {"id": 6, "key": "delta"}])

            result = query_field(tmp_path, "key", "alpha")
            self.assertEqual(len(result), 3)
            ids = sorted([r["id"] for r in result])
            self.assertEqual(ids, [1, 3, 5])


if __name__ == "__main__":
    unittest.main()
