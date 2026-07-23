import unittest

from training_data_packer.mode.release import parallel_package_pipeline
from training_data_packer.utils.metadata import Metadata


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

        result, metrics = parallel_package_pipeline(src_data, metadata, part_config, piis, contaminations)
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

        result, metrics = parallel_package_pipeline(src_data, metadata, part_config, piis, contaminations)
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

        result, metrics = parallel_package_pipeline(src_data, metadata, part_config, piis, contaminations)
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

        result, metrics = parallel_package_pipeline(src_data, metadata, part_config, piis, contaminations)
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

        result, metrics = parallel_package_pipeline(src_data, metadata, part_config, piis, contaminations)
        result = list(result)
        self.assertEqual(1, len(result))
        self.assertOnOf(expect_on_of, result)
