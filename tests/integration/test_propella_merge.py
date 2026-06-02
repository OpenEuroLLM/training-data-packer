import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from training_data_packer import propella_merge
from training_data_packer.propella_merge import process_file
from training_data_packer.utils.file import GenericJsonlReader, JsonlZstWriter
from training_data_packer.utils.metrics import read_metrics_from_file


class TestProcessFileIntegration(unittest.TestCase):
    def test_process_skip_existing_file(self):
        """
        Test that process_file skips files that already exist in the output directory.
        """
        with TemporaryDirectory() as test_dir:
            metadata = {"id": "id", "suffix": ".jsonl.zst"}
            source_name = "test_file.jsonl.zst"
            propella_dir = Path(test_dir).joinpath("propella-4b")
            propella_dir.mkdir(parents=True)

            output_file = propella_dir.joinpath(source_name)

            initial_data = [{"id": "initial", "value": 999}]
            JsonlZstWriter(output_file).write(iter(initial_data))

            process_file(metadata, source_name, propella_dir)

            result = list(GenericJsonlReader(output_file).read())
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], {"id": "initial", "value": 999})

    def test_process_structure(self):
        """
        Test merge two shards of propella data into one.
        """
        test_data = Path("tests/resources/integration/propella_merge")
        with TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir).joinpath("workdir")
            shutil.copytree(test_data, workdir)
            propella_merge.process(workdir, "part01")

            result = list(GenericJsonlReader(workdir.joinpath("propella-4b/part01/file_01.jsonl.zst")).read())
            self.assertEqual(4, len(result))
            self.assertEqual({"id": "id1"}, result[0])
            self.assertEqual({"id": "id2", "propella": "bar"}, result[1])
            self.assertEqual({"id": "id3", "propella": "foo"}, result[2])
            self.assertEqual({"id": "id4"}, result[3])

            metrics = read_metrics_from_file(workdir.joinpath("propella-4b/part01/.file_01.jsonl.zst.metrics.json"))
            self.assertEqual(
                {"processed_rows": 4, "rows_with_duplicates": 1, "rows_with_only_id": 1}, metrics["propella_merge"]
            )
            self.assertEqual(2, len(metrics["propella_sub"]))


if __name__ == "__main__":
    unittest.main()
