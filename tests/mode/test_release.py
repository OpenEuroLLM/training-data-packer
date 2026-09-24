import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from training_data_packer.metadata.metadata import Metadata
from training_data_packer.mode import release
from training_data_packer.mode.release import parallel_package_pipeline
from training_data_packer.utils.file import GenericJsonlReader


class IntegrationTests(unittest.TestCase):
    def test_flat_release(self):
        test_data = Path("tests/resources/integration/flat_release")
        with TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir).joinpath("workdir")
            shutil.copytree(test_data, workdir)
            out_dir = Path(workdir).joinpath("release-raw")
            release.process(workdir)

            source_file = list(GenericJsonlReader(workdir.joinpath("source/shard01/file_01.jsonl.zst")).read())
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

            with open(out_dir.joinpath("shard01/.file_01.jsonl.zst.metrics.json")) as file:
                metrics = json.load(file)
                self.assertEqual(
                    {
                        "input": {"lines_read": 5},
                        "pii_masker": {"masked_documents": 2, "pii_documents": 2},
                        "contamination": {"list_length": 2, "removed": 2},
                        "output": {"lines_written": 3},
                    },
                    metrics,
                )

    def test_block_list(self):
        test_data = Path("tests/resources/integration/block_list")
        with TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir).joinpath("workdir")
            shutil.copytree(test_data, workdir)
            out_dir = Path(workdir).joinpath("release-raw")
            release.process(test_data)

            source_file = list(GenericJsonlReader(workdir.joinpath("source/shard01/file_01.jsonl.zst")).read())
            result = list(GenericJsonlReader(out_dir.joinpath("shard01/file_01.jsonl.zst")).read())

            self.assertEqual(3, len(result))
            self.assertEqual(source_file[0]["id"], result[0]["id"])

            self.assertEqual(source_file[1], result[1])

            self.assertEqual(source_file[5]["id"], result[2]["id"])

            with open(out_dir.joinpath("shard01/.file_01.jsonl.zst.metrics.json")) as file:
                metrics = json.load(file)
                self.assertEqual(
                    {
                        "input": {"lines_read": 6},
                        "pii_masker": {"masked_documents": 0, "pii_documents": 0},
                        "block_list": {"list_length": 1, "removed": 1},
                        "contamination": {"list_length": 2, "removed": 2},
                        "output": {"lines_written": 3},
                    },
                    metrics,
                )


class TestParallelPackagePipeline(unittest.TestCase):
    def assertOnOf(self, expect_on_of, result):
        found = False
        for e in expect_on_of:
            if e == result[0]:
                found = True
                break
        self.assertTrue(found)

    def test_first(self):
        src_data = [
            {
                "source_text": "Antarctica is the coldest place on Earth.",
                "target_text": "Antarktida është kontinent më i ftohtë në tokë.",
                "src_lang": "eng_Latn",
                "tgt_lang": "als_Latn",
            }
        ]
        metadata = Metadata(
            {
                "_internal": {"parallel": True},
                "text": "text",
                "parallel": {
                    "input": "source",
                    "source": {
                        "text": "source_text",
                        "language": "src_lang",
                    },
                    "target": {
                        "text": "target_text",
                        "language": "tgt_lang",
                    },
                },
            }
        )
        part_config = {}
        piis = iter([])
        contaminations = iter([])
        expect_on_of = [
            {
                "id": "14882f09150ccfb56e01519d5913ae8c869d910c018d69b3b01eba884595473f",
                "text": "English: Antarctica is the coldest place on Earth.\n"
                "Tosk Albanian: Antarktida është kontinent më i ftohtë në tokë.",
            },
            {
                "id": "9b27af5e0a5e7ed3630448a0b04731b7a986d9f899873ffdff8694c7359896bb",
                "text": "Tosk Albanian: Antarktida është kontinent më i ftohtë në tokë.\n"
                "English: Antarctica is the coldest place on Earth.",
            },
        ]

        result, _ = parallel_package_pipeline(src_data, metadata, part_config, piis, contaminations)
        result = list(result)
        self.assertOnOf(expect_on_of, result)

    def test_pii_without_id(self):
        src_data = [
            {
                "source_text": "Shall we play a game?",
                "target_text": "Duam të luajmë?",
                "src_lang": "eng_Latn",
                "tgt_lang": "als_Latn",
            },
            {
                "source_text": "Antarctica is the coldest place on Earth.",
                "target_text": "Antarktida është kontinent më i ftohtë në tokë.",
                "src_lang": "eng_Latn",
                "tgt_lang": "als_Latn",
            },
        ]
        metadata = Metadata(
            {
                "_internal": {"parallel": True},
                "text": "text",
                "parallel": {
                    "input": "source",
                    "source": {
                        "text": "source_text",
                        "language": "src_lang",
                    },
                    "target": {
                        "text": "target_text",
                        "language": "tgt_lang",
                    },
                },
            }
        )
        part_config = {}
        piis = iter([{"hash": "7ea462d1386566606429b6ba9c3adfe39d742c742c835b464a8e8ccdfdf71452"}])
        contaminations = iter([])
        expect_on_of = [
            {
                "id": "03bcfae9b977c43b0061671d7141f318df4b975e447bdb4703439e9ec615aa89",
                "text": "English: Shall we play a game?\nTosk Albanian: Duam të luajmë?",
            },
            {
                "id": "bff982083ca3c3d9bd0e5c6396db9c14a4e34a05c9cffe3b07859d2a022fa2c0",
                "text": "Tosk Albanian: Duam të luajmë?\nEnglish: Shall we play a game?",
            },
        ]

        result, _ = parallel_package_pipeline(src_data, metadata, part_config, piis, contaminations)
        result = list(result)
        self.assertEqual(1, len(result))
        self.assertOnOf(expect_on_of, result)

    def test_pii_without_id_but_hash_as_id(self):
        src_data = [
            {
                "source_text": "Shall we play a game?",
                "target_text": "Duam të luajmë?",
                "src_lang": "eng_Latn",
                "tgt_lang": "als_Latn",
            },
            {
                "source_text": "Antarctica is the coldest place on Earth.",
                "target_text": "Antarktida është kontinent më i ftohtë në tokë.",
                "src_lang": "eng_Latn",
                "tgt_lang": "als_Latn",
            },
        ]
        metadata = {
            "_internal": {"parallel": True},
            "text": "text",
            "parallel": {
                "input": "source",
                "source": {
                    "text": "source_text",
                    "language": "src_lang",
                },
                "target": {
                    "text": "target_text",
                    "language": "tgt_lang",
                },
            },
        }
        part_config = {}
        piis = iter([{"id": "7ea462d1386566606429b6ba9c3adfe39d742c742c835b464a8e8ccdfdf71452"}])
        contaminations = iter([])
        expect_on_of = [
            {
                "id": "03bcfae9b977c43b0061671d7141f318df4b975e447bdb4703439e9ec615aa89",
                "text": "English: Shall we play a game?\nTosk Albanian: Duam të luajmë?",
            },
            {
                "id": "bff982083ca3c3d9bd0e5c6396db9c14a4e34a05c9cffe3b07859d2a022fa2c0",
                "text": "Tosk Albanian: Duam të luajmë?\nEnglish: Shall we play a game?",
            },
        ]

        result, _ = parallel_package_pipeline(src_data, metadata, part_config, piis, contaminations)
        result = list(result)
        self.assertEqual(1, len(result))
        self.assertOnOf(expect_on_of, result)

    def test_pii_with_id(self):
        src_data = [
            {
                "id": 1,
                "source_text": "Shall we play a game?",
                "target_text": "Duam të luajmë?",
                "src_lang": "eng_Latn",
                "tgt_lang": "als_Latn",
            },
            {
                "id": 2,
                "source_text": "Antarctica is the coldest place on Earth.",
                "target_text": "Antarktida është kontinent më i ftohtë në tokë.",
                "src_lang": "eng_Latn",
                "tgt_lang": "als_Latn",
            },
        ]
        metadata = Metadata(
            {
                "_internal": {"parallel": True},
                "id": "id",
                "text": "text",
                "parallel": {
                    "input": "source",
                    "source": {
                        "text": "source_text",
                        "language": "src_lang",
                    },
                    "target": {
                        "text": "target_text",
                        "language": "tgt_lang",
                    },
                },
            }
        )
        part_config = {}
        piis = iter([{"id": "2"}])
        contaminations = iter([])
        expect_on_of = [
            {
                "id": "03bcfae9b977c43b0061671d7141f318df4b975e447bdb4703439e9ec615aa89",
                "text": "English: Shall we play a game?\nTosk Albanian: Duam të luajmë?",
            },
            {
                "id": "bff982083ca3c3d9bd0e5c6396db9c14a4e34a05c9cffe3b07859d2a022fa2c0",
                "text": "Tosk Albanian: Duam të luajmë?\nEnglish: Shall we play a game?",
            },
        ]

        result, _ = parallel_package_pipeline(src_data, metadata, part_config, piis, contaminations)
        result = list(result)
        self.assertEqual(1, len(result))
        self.assertOnOf(expect_on_of, result)

    def test_contamination_without_id(self):
        src_data = [
            {
                "source_text": "Antarctica is the coldest place on Earth.",
                "target_text": "Antarktida është kontinent më i ftohtë në tokë.",
                "src_lang": "eng_Latn",
                "tgt_lang": "als_Latn",
            },
            {
                "source_text": "Shall we play a game?",
                "target_text": "Duam të luajmë?",
                "src_lang": "eng_Latn",
                "tgt_lang": "als_Latn",
            },
        ]
        metadata = Metadata(
            {
                "_internal": {"parallel": True},
                "text": "text",
                "parallel": {
                    "input": "source",
                    "source": {
                        "text": "source_text",
                        "language": "src_lang",
                    },
                    "target": {
                        "text": "target_text",
                        "language": "tgt_lang",
                    },
                },
            }
        )
        part_config = {}
        piis = iter([])
        contaminations = iter([{"hash": "7ea462d1386566606429b6ba9c3adfe39d742c742c835b464a8e8ccdfdf71452"}])
        expect_on_of = [
            {
                "id": "03bcfae9b977c43b0061671d7141f318df4b975e447bdb4703439e9ec615aa89",
                "text": "English: Shall we play a game?\nTosk Albanian: Duam të luajmë?",
            },
            {
                "id": "bff982083ca3c3d9bd0e5c6396db9c14a4e34a05c9cffe3b07859d2a022fa2c0",
                "text": "Tosk Albanian: Duam të luajmë?\nEnglish: Shall we play a game?",
            },
        ]

        result, _ = parallel_package_pipeline(src_data, metadata, part_config, piis, contaminations)
        result = list(result)
        self.assertEqual(1, len(result))
        self.assertOnOf(expect_on_of, result)

    def test_contamination_with_id(self):
        src_data = [
            {
                "id": "1",
                "source_text": "Antarctica is the coldest place on Earth.",
                "target_text": "Antarktida është kontinent më i ftohtë në tokë.",
                "src_lang": "eng_Latn",
                "tgt_lang": "als_Latn",
            },
            {
                "id": "2",
                "source_text": "Shall we play a game?",
                "target_text": "Duam të luajmë?",
                "src_lang": "eng_Latn",
                "tgt_lang": "als_Latn",
            },
        ]
        metadata = Metadata(
            {
                "_internal": {"parallel": True},
                "id": "id",
                "text": "text",
                "parallel": {
                    "input": "source",
                    "source": {
                        "text": "source_text",
                        "language": "src_lang",
                    },
                    "target": {
                        "text": "target_text",
                        "language": "tgt_lang",
                    },
                },
            }
        )
        part_config = {}
        piis = iter([])
        contaminations = iter([{"id": "1"}])
        expect_on_of = [
            {
                "id": "03bcfae9b977c43b0061671d7141f318df4b975e447bdb4703439e9ec615aa89",
                "text": "English: Shall we play a game?\nTosk Albanian: Duam të luajmë?",
            },
            {
                "id": "bff982083ca3c3d9bd0e5c6396db9c14a4e34a05c9cffe3b07859d2a022fa2c0",
                "text": "Tosk Albanian: Duam të luajmë?\nEnglish: Shall we play a game?",
            },
        ]

        result, _ = parallel_package_pipeline(src_data, metadata, part_config, piis, contaminations)
        result = list(result)
        self.assertEqual(1, len(result))
        self.assertOnOf(expect_on_of, result)
