import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

from training_data_packer.collect_metrics import main


class TestCollectMetricsTree(unittest.TestCase):
    def test_collect_metrics_tree(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create metadata.yaml
            metadata = {
                "release": {"default": {"pack": "tree"}, "part1": {}, "part2": {}},
                "source": {"part1": {}, "part2": {}},
            }
            with open(tmp_path / "metadata.yaml", "w") as f:
                yaml.dump(metadata, f)

            # Create release_raw structure
            release_raw = tmp_path / "release_raw"
            release_raw.mkdir()

            part1_dir = release_raw / "part1"
            part1_dir.mkdir()
            m1 = {"input": {"lines_read": 10}, "output": {"lines_written": 8}}
            (part1_dir / ".f1.metrics.json").write_text(json.dumps(m1))

            part2_dir = release_raw / "part2"
            part2_dir.mkdir()
            m2 = {"input": {"lines_read": 20}, "output": {"lines_written": 15}}
            (part2_dir / ".f2.metrics.json").write_text(json.dumps(m2))

            # Run main
            with patch.object(sys, "argv", ["oellm-collect-metrics", "--collection-dir", str(tmp_path)]):
                main()

            # Check results
            self.assertTrue((part1_dir / "metrics.json").exists())
            self.assertTrue((part2_dir / "metrics.json").exists())
            self.assertTrue((release_raw / "metrics.json").exists())

            with open(part1_dir / "metrics.json") as f:
                res1 = json.load(f)
                self.assertEqual(res1["input"]["lines_read"], 10)

            with open(part2_dir / "metrics.json") as f:
                res2 = json.load(f)
                self.assertEqual(res2["input"]["lines_read"], 20)


if __name__ == "__main__":
    unittest.main()
