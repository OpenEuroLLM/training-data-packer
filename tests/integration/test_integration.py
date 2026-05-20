import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from training_data_packer.app import process
from training_data_packer.utils.file import GenericJsonlReader


class IntegrationTests(unittest.TestCase):
    def test_flat_release(self):
        test_data = Path("tests/resources/integration/flat_release")
        with TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir).joinpath("output")
            process(test_data, out_dir)

            source_file = list(GenericJsonlReader(test_data.joinpath("source/shard01/file_01.jsonl.zst")).read())
            result = list(GenericJsonlReader(out_dir.joinpath("shard01/file_01.jsonl.zst")).read())

            self.assertEqual(3, len(result))
            self.assertEqual(source_file[0]["id"], result[0]["id"])
            self.assertNotEqual(source_file[0]["text"], result[0]["text"])
            self.assertEqual(1, result[0]["pii_masks"])

            self.assertEqual(source_file[1], result[1])
            self.assertTrue("pii_masks" not in result[1])

            self.assertEqual(source_file[4]["id"], result[2]["id"])
            self.assertNotEqual(source_file[4]["text"], result[2]["text"])
            self.assertEqual(2, result[2]["pii_masks"])

            with open(out_dir.joinpath("shard01/.file_01.jsonl.zst.metrics.json"), mode="r") as file:
                metrics = json.load(file)
                self.assertEqual(
                    {
                        "input": {"lines_read": 5},
                        "contamination": {"list_length": 2, "removed": 2},
                        "output": {"lines_written": 3},
                    },
                    metrics
                )

    def test_block_list(self):
        test_data = Path("tests/resources/integration/block_list")
        with TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir).joinpath("output")
            process(test_data, out_dir)

            source_file = list(GenericJsonlReader(test_data.joinpath("source/shard01/file_01.jsonl.zst")).read())
            result = list(GenericJsonlReader(out_dir.joinpath("shard01/file_01.jsonl.zst")).read())

            self.assertEqual(3, len(result))
            self.assertEqual(source_file[0]["id"], result[0]["id"])

            self.assertEqual(source_file[1], result[1])

            self.assertEqual(source_file[5]["id"], result[2]["id"])

            with open(out_dir.joinpath("shard01/.file_01.jsonl.zst.metrics.json"), mode="r") as file:
                metrics = json.load(file)
                self.assertEqual(
                    {
                        "input": {"lines_read": 6},
                        "block_list": {"list_length": 1, "removed": 1},
                        "contamination": {"list_length": 2, "removed": 2},
                        "output": {"lines_written": 3},
                    },
                    metrics,
                )


if __name__ == "__main__":
    unittest.main()
