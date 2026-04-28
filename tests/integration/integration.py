import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from training_data_packer.app import main


class IntegrationTests(unittest.TestCase):
    def test_non_partitioned_data(self):
        test_data = Path("tests/resources/integration/non_partitioned")
        with TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir).joinpath("output")
            main(test_data, out_dir)
            result = subprocess.run(["diff", test_data.joinpath("expected/file_01.jsonl.zst"), out_dir.joinpath("file_01.jsonl.zst")], capture_output=True)
            print(result.stdout)
            self.assertEqual(0, result.returncode)


if __name__ == '__main__':
    unittest.main()
