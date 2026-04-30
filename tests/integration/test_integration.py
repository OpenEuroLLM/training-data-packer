import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from training_data_packer.app import process
from training_data_packer.jsonl_zst import JsonlZstReader


class IntegrationTests(unittest.TestCase):
    def test_non_partitioned_data(self):
        test_data = Path("tests/resources/integration/non_partitioned")
        with TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir).joinpath("output")
            process(test_data, out_dir)

            source_file = list(JsonlZstReader(test_data.joinpath("source/file_01.jsonl.zst")).read())
            result = list(JsonlZstReader(out_dir.joinpath("file_01.jsonl.zst")).read())

            self.assertEqual(len(result), 3)
            self.assertEqual(result[0]["id"], source_file[0]["id"])
            self.assertNotEqual(result[0]["text"], source_file[0]["text"])
            self.assertEqual(result[0]["pii_masks"], 1)

            self.assertEqual(result[1], source_file[1])
            self.assertTrue("pii_masks" not in result[1])

            self.assertEqual(result[2]["id"], source_file[4]["id"])
            self.assertNotEqual(result[2]["text"], source_file[4]["text"])
            self.assertEqual(result[2]["pii_masks"], 2)


if __name__ == '__main__':
    unittest.main()
