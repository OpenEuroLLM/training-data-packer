import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from training_data_packer.utils import metrics


class DummyClassWithMetrics:
    def __init__(self, test_metrics: dict[str, Any]):
        self.my_metrics = test_metrics

    def get_metrics(self) -> dict[str, Any]:
        return self.my_metrics


class TestMetrics(unittest.TestCase):
    def test_collect_metrics_with_only_nones(self):
        result = metrics.collect_metrics(None, None)
        self.assertEqual(result, {})

    def test_collect_metrics_with_objects(self):
        object1 = DummyClassWithMetrics({"foo": "bar"})
        object2 = DummyClassWithMetrics({"gazonk": {"lines": 5}})
        result = metrics.collect_metrics(object1, None, object2)
        self.assertEqual(result, {"foo": "bar", "gazonk": {"lines": 5}})


class TestMetricsAggregation(unittest.TestCase):
    def test_aggregate_metrics(self):
        m1 = {"input": {"lines_read": 10}, "output": {"lines_written": 8, "str_value": "value"}}
        m2 = {"input": {"lines_read": 20}, "output": {"lines_written": 15, "str_value": "value"}}

        result = metrics.aggregate_metrics([m1, m2])

        expected = {"input": {"lines_read": 30}, "output": {"lines_written": 23, "str_value": "value"}}
        self.assertEqual(result, expected)

    def test_aggregate_metrics_with_missing_keys(self):
        m1 = {
            "input": {"lines_read": 10},
        }
        m2 = {"output": {"lines_written": 15}}

        result = metrics.aggregate_metrics([m1, m2])

        expected = {"input": {"lines_read": 10}, "output": {"lines_written": 15}}
        self.assertEqual(result, expected)


class TestCollectMetricsCLI(unittest.TestCase):
    def test_collect_metrics_from_dir(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create some dummy metrics files
            m1 = {"input": {"lines_read": 10}, "output": {"lines_written": 8}}
            m2 = {"input": {"lines_read": 20}, "output": {"lines_written": 15}}

            # File names should start with . and end with .metrics.json
            (tmp_path / ".file1.jsonl.zst.metrics.json").write_text(json.dumps(m1))
            (tmp_path / "subdir").mkdir()
            (tmp_path / "subdir" / ".file2.jsonl.zst.metrics.json").write_text(json.dumps(m2))

            # This one should be ignored (doesn't start with .)
            (tmp_path / "other.metrics.json").write_text(json.dumps({"should": "ignore"}))

            metrics.merge_metrics(tmp_path)

            output_file = tmp_path / "metrics.json"
            self.assertTrue(output_file.exists())

            with open(output_file) as f:
                result = json.load(f)

            expected = {"input": {"lines_read": 30}, "output": {"lines_written": 23}}
            self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
