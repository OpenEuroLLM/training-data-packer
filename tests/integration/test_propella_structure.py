import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import orjson as json
import zstandard as zstd

from training_data_packer.propella_structure import process
from training_data_packer.utils.file import GenericJsonlReader


class TestPropellaStructure(unittest.TestCase):
    def test_propella_structure_basic(self):
        """
        Test basic propella structure processing.
        Source has 4 records (id1, id2, id3, id4).
        Propella has records for id1, id2, id4.
        id3 is not in propella and should be written as {"id": "id3"}.
        """
        test_data = Path("tests/resources/integration/propella_structure")
        with TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir).joinpath("workdir")
            shutil.copytree(test_data, workdir)

            process(workdir, workdir.joinpath("data.parquet").parent)

            output_file = workdir.joinpath("propella-4b/shard01/file_01.jsonl.zst")
            self.assertTrue(output_file.exists())

            result = list(GenericJsonlReader(output_file).read())
            self.assertEqual(4, len(result))

            id1_record = next((r for r in result if r["id"] == "id1"), None)
            self.assertIsNotNone(id1_record)
            self.assertIn("metadata", id1_record["propella-4b"])
            self.assertEqual({"key": "value1"}, id1_record["propella-4b"]["metadata"])

            id2_record = next((r for r in result if r["id"] == "id2"), None)
            self.assertIsNotNone(id2_record)
            self.assertIn("metadata", id2_record["propella-4b"])
            self.assertEqual({"key": "value2"}, id2_record["propella-4b"]["metadata"])

            id3_record = next((r for r in result if r["id"] == "id3"), None)
            self.assertIsNotNone(id3_record)
            self.assertEqual(["id"], list(id3_record.keys()))

            id4_record = next((r for r in result if r["id"] == "id4"), None)
            self.assertIsNotNone(id4_record)
            self.assertIn("metadata", id4_record["propella-4b"])
            self.assertEqual({"key": "value4"}, id4_record["propella-4b"]["metadata"])

    def test_propella_structure_skip_existing(self):
        """
        Test that existing output files are skipped.
        """
        test_data = Path("tests/resources/integration/propella_structure")
        with TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir).joinpath("workdir")
            shutil.copytree(test_data, workdir)

            output_file = workdir.joinpath("propella/shard01/file_01.jsonl.zst")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            existing_data = [{"id": "existing"}]
            cctx = zstd.ZstdCompressor()
            with open(output_file, "wb") as f:
                with cctx.stream_writer(f) as compressor:
                    for record in existing_data:
                        compressor.write(json.dumps(record))
                        compressor.write(b"\n")

            process(workdir, workdir.joinpath("data.parquet").parent)

            result = list(GenericJsonlReader(output_file).read())
            self.assertEqual(1, len(result))
            self.assertEqual("existing", result[0]["id"])

    def test_propella_structure_deduplicated_ids(self):
        """
        Test that duplicate IDs in source are handled correctly.
        Each record is processed independently, so duplicates should produce duplicates.
        """
        test_data = Path("tests/resources/integration/propella_structure")
        with TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir).joinpath("workdir")
            shutil.copytree(test_data, workdir)

            source_file = workdir.joinpath("source/shard01/file_02.jsonl.zst")
            source_file.parent.mkdir(parents=True, exist_ok=True)
            cctx = zstd.ZstdCompressor()
            with open(source_file, "wb") as f:
                with cctx.stream_writer(f) as compressor:
                    for record in [{"id": "id1"}, {"id": "id1"}, {"id": "id2"}, {"id": "id3"}]:
                        compressor.write(json.dumps(record))
                        compressor.write(b"\n")

            process(workdir, workdir.joinpath("data.parquet").parent)

            output_file = workdir.joinpath("propella-4b/shard01/file_02.jsonl.zst")
            self.assertTrue(output_file.exists())

            result = list(GenericJsonlReader(output_file).read())
            self.assertEqual(4, len(result))

            id1_records = [r for r in result if r["id"] == "id1"]
            self.assertEqual(2, len(id1_records))

            id2_record = next((r for r in result if r["id"] == "id2"), None)
            self.assertIsNotNone(id2_record)
            self.assertEqual({"key": "value2"}, id2_record["propella-4b"]["metadata"])

            id3_record = next((r for r in result if r["id"] == "id3"), None)
            self.assertIsNotNone(id3_record)
            self.assertEqual(["id"], list(id3_record.keys()))


if __name__ == "__main__":
    unittest.main()
