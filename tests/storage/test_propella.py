import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
from parameterized import parameterized

from training_data_packer.storage.propella import get_lookup_fn


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


class TestGetLookupFn(unittest.TestCase):
    def test_lookup_with_matching_records(self):
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

            lookup_fn = get_lookup_fn(tmp_path, "status")
            result = lookup_fn("active")
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

            lookup_fn = get_lookup_fn(tmp_path, "status")
            result = lookup_fn("inactive")
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

            lookup_fn = get_lookup_fn(tmp_path, "score")
            result = lookup_fn(85)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["name"], "Alice")
            self.assertEqual(result[1]["name"], "Charlie")

    @parameterized.expand(
        [
            ["dash", "field-name"],
            ["space", "field name"],
            ["dot", "field.name"],
            ["special", "field@name"],
        ]
    )
    def test_invalid_field_name(self, name, field_name):
        with TemporaryDirectory() as tmp_path:
            with self.assertRaises(ValueError) as cm:
                get_lookup_fn(tmp_path, field_name)
            self.assertIn("only alphanumeric characters and underscores are allowed", str(cm.exception))

    def test_valid_field_name_alphanumeric_only(self):
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

            lookup_fn = get_lookup_fn(tmp_path, "abc123")
            result = lookup_fn("value1")
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["id"], 1)

    def test_valid_field_name_with_underscores(self):
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

            lookup_fn = get_lookup_fn(tmp_path, "field_name")
            result = lookup_fn("value1")
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["id"], 1)

    def test_not_a_directory(self):
        with TemporaryDirectory() as tmp_path:
            test_file = Path(tmp_path) / "test.parquet"
            test_file.write_text("not a directory")
            with self.assertRaises(NotADirectoryError):
                get_lookup_fn(test_file, "field_name")

    def test_no_parquet_files(self):
        with TemporaryDirectory() as tmp_path:
            with self.assertRaises(FileNotFoundError) as cm:
                get_lookup_fn(tmp_path, "field_name")
            self.assertIn("No parquet files found", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
