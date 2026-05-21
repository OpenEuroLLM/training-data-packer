import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from training_data_packer.collect_metrics import collect_metrics


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

            collect_metrics(tmp_path)

            output_file = tmp_path / "metrics.json"
            self.assertTrue(output_file.exists())

            with open(output_file) as f:
                result = json.load(f)

            expected = {"input": {"lines_read": 30}, "output": {"lines_written": 23}}
            self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
